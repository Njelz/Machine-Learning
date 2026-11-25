import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lab03" / "src"))

from models import CNN


SEED = 42
EFFECTIVE_BATCH_SIZE = 256
MICRO_BATCH_SIZE = 64
ACCUMULATION_STEPS = 4


def evaluate_accuracy(model, loader, device):
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    device,
    use_accumulation,
):
    model.train()
    total_loss = 0.0
    total_examples = 0

    if not use_accumulation:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

    else:
        iterator = iter(loader)

        while True:
            # Collect up to four micro-batches for one optimizer update.
            group = []
            for _ in range(ACCUMULATION_STEPS):
                try:
                    group.append(next(iterator))
                except StopIteration:
                    break

            if not group:
                break

            group_size = sum(labels.size(0) for _, labels in group)
            optimizer.zero_grad()

            for images, labels in group:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)

                # Sum the per-example losses, then scale by the
                # number of examples represented in this update.
                loss_sum = nn.functional.cross_entropy(
                    logits,
                    labels,
                    reduction="sum",
                )
                (loss_sum / group_size).backward()

                total_loss += loss_sum.item()
                total_examples += labels.size(0)

            optimizer.step()

    return total_loss / total_examples


def run_experiment(
    mode,
    train_set,
    validation_loader,
    test_loader,
    config,
    device,
):
    # Reset seeds so both runs start from the same weights and data order.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    batch_size = (
        EFFECTIVE_BATCH_SIZE
        if mode == "direct"
        else MICRO_BATCH_SIZE
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        generator=torch.Generator().manual_seed(SEED),
    )

    model = CNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_function = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            use_accumulation=(mode == "accumulation"),
        )
        validation_accuracy = evaluate_accuracy(
            model,
            validation_loader,
            device,
        )

        history.append({
            "mode": mode,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_accuracy": validation_accuracy,
        })

        print(
            f"{mode} epoch {epoch}: "
            f"train_loss={train_loss:.5f}, "
            f"validation_accuracy={validation_accuracy:.4f}"
        )

    test_accuracy = evaluate_accuracy(model, test_loader, device)

    return history, test_accuracy


def main():
    config_path = REPO_ROOT / "lab06" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
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

    train_set, validation_set = random_split(
        full_train_set,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    test_set = datasets.FashionMNIST(
        root=REPO_ROOT / config["dataset"]["root"],
        train=False,
        download=True,
        transform=transform,
    )

    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]

    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(
        f"Direct batch size: {EFFECTIVE_BATCH_SIZE}; "
        f"accumulation: {MICRO_BATCH_SIZE} × {ACCUMULATION_STEPS}; "
        f"epochs: {config['training']['epochs']}"
    )

    direct_history, direct_test_accuracy = run_experiment(
        "direct",
        train_set,
        validation_loader,
        test_loader,
        config,
        device,
    )

    accumulation_history, accumulation_test_accuracy = run_experiment(
        "accumulation",
        train_set,
        validation_loader,
        test_loader,
        config,
        device,
    )

    results_dir = REPO_ROOT / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_history = direct_history + accumulation_history

    csv_path = results_dir / "gradient_accumulation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "mode",
            "epoch",
            "train_loss",
            "validation_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_history)

    summary_path = results_dir / "gradient_accumulation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["mode", "test_accuracy"])
        writer.writerow(["direct_batch_256", direct_test_accuracy])
        writer.writerow(["batch_64_accumulation_4", accumulation_test_accuracy])

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    for mode, rows in [
        ("direct batch 256", direct_history),
        ("batch 64, accumulate 4", accumulation_history),
    ]:
        axes[0].plot(
            [row["epoch"] for row in rows],
            [row["train_loss"] for row in rows],
            marker="o",
            label=mode,
        )
        axes[1].plot(
            [row["epoch"] for row in rows],
            [row["validation_accuracy"] for row in rows],
            marker="o",
            label=mode,
        )

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    plot_path = results_dir / "gradient_accumulation.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print("\nFinal test accuracy:")
    print(f"Direct batch 256: {direct_test_accuracy:.4f}")
    print(f"Batch 64, accumulate 4: {accumulation_test_accuracy:.4f}")
    print(f"\nSaved curves: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()