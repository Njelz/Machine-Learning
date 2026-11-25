import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lab03" / "src"))

from models import CNN


EPOCHS = 10
OPTIMIZERS = ["Adam", "SGD_momentum"]
SCHEDULERS = ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def make_optimizer(name, model, learning_rate, weight_decay):
    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if name == "SGD_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(name, optimizer):
    if name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1,
        )

    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
        )

    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
        )

    raise ValueError(f"Unknown scheduler: {name}")


def main():
    config_path = REPO_ROOT / "lab06" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    device = torch.device(config["training"]["device"])
    data_root = REPO_ROOT / config["dataset"]["root"]

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

    loss_function = nn.CrossEntropyLoss()
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment("lab06_optimizer_scheduler_comparison")

    epoch_rows = []
    summary_rows = []

    for optimizer_name in OPTIMIZERS:
        for scheduler_name in SCHEDULERS:
            run_seed = seed
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(run_seed),
            )

            model = CNN().to(device)
            optimizer = make_optimizer(
                optimizer_name,
                model,
                learning_rate,
                weight_decay,
            )
            scheduler = make_scheduler(scheduler_name, optimizer)

            run_name = f"{optimizer_name}_{scheduler_name}"
            print(f"\nRunning {run_name} for {EPOCHS} epochs")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "momentum": 0.9 if optimizer_name == "SGD_momentum" else 0,
                    "batch_size": batch_size,
                    "epochs": EPOCHS,
                    "num_workers": num_workers,
                    "seed": run_seed,
                    "step_lr_step_size": 3,
                    "step_lr_gamma": 0.1,
                    "cosine_t_max": EPOCHS,
                    "plateau_factor": 0.1,
                    "plateau_patience": 2,
                })

                for epoch in range(1, EPOCHS + 1):
                    model.train()
                    total_train_loss = 0.0
                    total_train_examples = 0

                    for images, labels in train_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()
                        logits = model(images)
                        loss = loss_function(logits, labels)
                        loss.backward()
                        optimizer.step()

                        total_train_loss += loss.item() * labels.size(0)
                        total_train_examples += labels.size(0)

                    train_loss = total_train_loss / total_train_examples
                    validation_loss, validation_accuracy = evaluate(
                        model,
                        validation_loader,
                        loss_function,
                        device,
                    )

                    if scheduler_name == "ReduceLROnPlateau":
                        scheduler.step(validation_loss)
                    else:
                        scheduler.step()

                    current_lr = optimizer.param_groups[0]["lr"]

                    row = {
                        "optimizer": optimizer_name,
                        "scheduler": scheduler_name,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                        "learning_rate": current_lr,
                    }
                    epoch_rows.append(row)

                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy,
                        "learning_rate": current_lr,
                    }, step=epoch)

                    print(
                        f"Epoch {epoch}/{EPOCHS}: "
                        f"train_loss={train_loss:.4f}, "
                        f"val_loss={validation_loss:.4f}, "
                        f"val_accuracy={validation_accuracy:.4f}, "
                        f"lr={current_lr:.6g}"
                    )

                test_loss, test_accuracy = evaluate(
                    model,
                    test_loader,
                    loss_function,
                    device,
                )
                mlflow.log_metrics({
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                })

                summary_rows.append({
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                })

                print(
                    f"{run_name}: test_accuracy={test_accuracy:.4f}"
                )

    results_dir = REPO_ROOT / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    curves_csv = results_dir / "optimizer_scheduler_curves.csv"
    with curves_csv.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "optimizer",
            "scheduler",
            "epoch",
            "train_loss",
            "validation_loss",
            "validation_accuracy",
            "learning_rate",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(epoch_rows)

    summary_csv = results_dir / "optimizer_scheduler_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        fields = [
            "optimizer",
            "scheduler",
            "test_loss",
            "test_accuracy",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for axis, optimizer_name in zip(axes, OPTIMIZERS):
        for scheduler_name in SCHEDULERS:
            selected = [
                row for row in epoch_rows
                if row["optimizer"] == optimizer_name
                and row["scheduler"] == scheduler_name
            ]
            axis.plot(
                [row["epoch"] for row in selected],
                [row["validation_accuracy"] for row in selected],
                marker="o",
                label=scheduler_name,
            )

        axis.set_title(f"{optimizer_name}: validation accuracy")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation accuracy")
        axis.set_xticks(range(1, EPOCHS + 1))
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    plot_path = results_dir / "optimizer_scheduler_accuracy.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"\nSaved epoch metrics to: {curves_csv}")
    print(f"Saved test summary to: {summary_csv}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()