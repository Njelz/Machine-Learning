import argparse
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


def save_history(history, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "resume_training.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["epoch", "train_loss", "validation_accuracy"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    plot_path = results_dir / "resume_training_curves.png"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    accuracies = [row["validation_accuracy"] for row in history]

    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, accuracies, marker="o")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the saved checkpoint",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="pause and save after this epoch",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / "lab06" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["project"]["seed"]
    total_epochs = config["training"]["epochs"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

    device = torch.device(config["training"]["device"])
    data_root = REPO_ROOT / config["dataset"]["root"]
    checkpoint_dir = REPO_ROOT / config["checkpoint"]["directory"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "resume_cnn.pt"
    results_dir = REPO_ROOT / "lab06" / "results"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (config["dataset"]["normalize_mean"],),
            (config["dataset"]["normalize_std"],),
        ),
    ])

    full_train_set = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    validation_size = config["dataset"]["validation_size"]
    train_size = len(full_train_set) - validation_size

    train_set, validation_set = random_split(
        full_train_set,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

    test_set = datasets.FashionMNIST(
        root=data_root,
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

    model = CNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_function = nn.CrossEntropyLoss()

    start_epoch = 0
    history = []

    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}. "
                "Run once with --stop-after 3 first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]

        print(f"Resuming from checkpoint after epoch {start_epoch}.")
    else:
        print("Starting a new run.")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        # A repeatable shuffle order for each epoch also makes resuming consistent.
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed + epoch),
        )

        model.train()
        total_loss = 0.0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_examples += labels.size(0)

        train_loss = total_loss / total_examples
        validation_accuracy = evaluate(model, validation_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_accuracy": validation_accuracy,
        })

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )
        save_history(history, results_dir)

        print(
            f"Epoch {epoch}/{total_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"validation_accuracy={validation_accuracy:.4f}"
        )
        print(f"Checkpoint saved after epoch {epoch}.")

        if args.stop_after is not None and epoch >= args.stop_after:
            if epoch < total_epochs:
                print(
                    f"Paused at epoch {epoch}. "
                    "Run again with --resume to continue."
                )
                return

    if history and history[-1]["epoch"] == total_epochs:
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Training complete. Test accuracy: {test_accuracy:.4f}")
        print(f"Curves saved in: {results_dir}")


if __name__ == "__main__":
    main()