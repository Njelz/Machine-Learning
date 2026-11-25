import csv
import random
import sys
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lab03" / "src"))

from models import CNN


class RSSSampler:
    """Sample this process's resident memory while training."""

    def __init__(self, interval_seconds=0.05):
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop_event.is_set():
            try:
                memory_bytes = self.process.memory_info().rss
                self.peak_bytes = max(self.peak_bytes, memory_bytes)
            except psutil.Error:
                pass
            self._stop_event.wait(self.interval_seconds)

    def start(self):
        self.peak_bytes = self.process.memory_info().rss
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self.peak_bytes = max(
            self.peak_bytes,
            self.process.memory_info().rss,
        )


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def benchmark_one_batch_size(
    batch_size,
    train_set,
    test_set,
    config,
    run_seed,
    device,
):
    torch.manual_seed(run_seed)
    random.seed(run_seed)
    np.random.seed(run_seed)

    num_workers = config["dataset"]["num_workers"]

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(run_seed),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = CNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_function = nn.CrossEntropyLoss()

    sampler = RSSSampler()
    sampler.start()

    start_time = time.perf_counter()
    total_training_samples = 0

    for epoch in range(config["training"]["epochs"]):
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_training_samples += labels.size(0)

        print(f"  batch_size={batch_size}: finished epoch {epoch + 1}")

    training_seconds = time.perf_counter() - start_time
    sampler.stop()

    throughput = total_training_samples / training_seconds
    test_accuracy = evaluate(model, test_loader, device)
    peak_rss_mb = sampler.peak_bytes / (1024 ** 2)

    return {
        "batch_size": batch_size,
        "epochs": config["training"]["epochs"],
        "throughput_samples_per_second": throughput,
        "test_accuracy": test_accuracy,
        "peak_process_rss_mb": peak_rss_mb,
    }


def main():
    config_path = REPO_ROOT / "lab06" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

    device = torch.device(config["training"]["device"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (config["dataset"]["normalize_mean"],),
            (config["dataset"]["normalize_std"],),
        ),
    ])

    full_train_set = datasets.FashionMNIST(
        root=REPO_ROOT / config["dataset"]["root"],
        train=True,
        download=True,
        transform=transform,
    )

    validation_size = config["dataset"]["validation_size"]
    train_size = len(full_train_set) - validation_size

    train_set, _ = random_split(
        full_train_set,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

    test_set = datasets.FashionMNIST(
        root=REPO_ROOT / config["dataset"]["root"],
        train=False,
        download=True,
        transform=transform,
    )

    batch_sizes = [32, 64, 128, 256, 512]
    results = []

    print(f"Device: {device}; PyTorch CPU threads: {torch.get_num_threads()}")
    print(f"Training examples: {len(train_set)}")
    print(f"Epochs per batch size: {config['training']['epochs']}")
    print(f"Fixed num_workers: {config['dataset']['num_workers']}")

    for batch_size in batch_sizes:
        print(f"\nTesting batch size {batch_size}...")

        result = benchmark_one_batch_size(
            batch_size=batch_size,
            train_set=train_set,
            test_set=test_set,
            config=config,
            run_seed=seed,
            device=device,
        )
        results.append(result)

        print(
            f"Throughput: "
            f"{result['throughput_samples_per_second']:.1f} samples/second"
        )
        print(f"Test accuracy: {result['test_accuracy']:.4f}")
        print(
            f"Peak process RSS: "
            f"{result['peak_process_rss_mb']:.1f} MB"
        )

    results_dir = REPO_ROOT / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "batch_size_study.csv"
    fieldnames = [
        "batch_size",
        "epochs",
        "throughput_samples_per_second",
        "test_accuracy",
        "peak_process_rss_mb",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    plot_path = results_dir / "batch_size_study.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(
        [row["batch_size"] for row in results],
        [row["throughput_samples_per_second"] for row in results],
        marker="o",
    )
    axes[0].set_title("Training throughput")
    axes[0].set_xlabel("Batch size")
    axes[0].set_ylabel("Samples per second")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        [row["batch_size"] for row in results],
        [row["test_accuracy"] for row in results],
        marker="o",
    )
    axes[1].set_title("Test accuracy after 5 epochs")
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        [row["batch_size"] for row in results],
        [row["peak_process_rss_mb"] for row in results],
        marker="o",
    )
    axes[2].set_title("Peak process memory")
    axes[2].set_xlabel("Batch size")
    axes[2].set_ylabel("Peak RSS (MB)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"\nSaved results: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()