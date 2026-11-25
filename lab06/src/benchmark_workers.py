import csv
import gc
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]


def measure_one_pass(loader):
    samples = 0
    start = time.perf_counter()

    for _, labels in loader:
        samples += len(labels)

    elapsed = time.perf_counter() - start
    throughput = samples / elapsed
    return elapsed, throughput


def main():
    config_path = REPO_ROOT / "lab06" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["project"]["seed"]
    torch.manual_seed(seed)

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
    train_set, _ = random_split(
        full_train_set,
        [len(full_train_set) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

    batch_size = config["dataset"]["batch_size"]
    worker_counts = [0, 1, 2, 4, 8]
    results = []

    for worker_count in worker_counts:
        print(f"\nTesting num_workers={worker_count}")

        loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=worker_count,
            persistent_workers=(worker_count > 0),
            generator=torch.Generator().manual_seed(seed),
        )

        # One untimed pass warms up the loader and starts worker processes.
        measure_one_pass(loader)

        # Measure three more full passes and use their median.
        timings = [measure_one_pass(loader) for _ in range(3)]
        median_seconds = statistics.median(t[0] for t in timings)
        median_throughput = statistics.median(t[1] for t in timings)

        results.append({
            "num_workers": worker_count,
            "median_epoch_seconds": median_seconds,
            "median_samples_per_second": median_throughput,
        })

        print(f"Median pass time: {median_seconds:.2f} seconds")
        print(f"Median throughput: {median_throughput:.1f} samples/second")

        del loader
        gc.collect()

    results_dir = REPO_ROOT / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "worker_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "num_workers",
                "median_epoch_seconds",
                "median_samples_per_second",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    plot_path = results_dir / "worker_throughput.png"
    plt.figure(figsize=(7, 4))
    plt.plot(
        [row["num_workers"] for row in results],
        [row["median_samples_per_second"] for row in results],
        marker="o",
    )
    plt.xlabel("DataLoader num_workers")
    plt.ylabel("Median throughput (samples/second)")
    plt.title("Fashion-MNIST data-loading throughput")
    plt.xticks(worker_counts)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    print(f"\nSaved results to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()