import csv
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB03_SRC = REPO_ROOT / "lab03" / "src"
sys.path.insert(0, str(LAB03_SRC))

from models import CNN


def flatten_config(config, prefix=""):
    """Convert nested config values to MLflow-friendly key/value pairs."""
    flat = {}

    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, name))
        else:
            flat[name] = str(value)

    return flat


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    config_path = REPO_ROOT / "lab06" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(config["training"]["device"])
    data_root = REPO_ROOT / config["dataset"]["root"]
    results_dir = REPO_ROOT / "lab06" / "results"
    checkpoint_dir = REPO_ROOT / config["checkpoint"]["directory"]
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    train_size = len(full_train_set) - config["dataset"]["validation_size"]
    validation_size = config["dataset"]["validation_size"]

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
    loader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
    )
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
    loss_function = nn.CrossEntropyLoss()

    if config["training"]["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        raise ValueError(
            "This first pipeline version supports the Adam optimizer."
        )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    checkpoint_path = (
        checkpoint_dir / config["checkpoint"]["filename"]
    )
    metrics_path = results_dir / "training_metrics.csv"

    history = []
    best_validation_accuracy = -1.0

    with mlflow.start_run(run_name=config["project"]["name"]):
        mlflow.log_params(flatten_config(config))
        mlflow.log_artifact(str(config_path), artifact_path="config")

        for epoch in range(1, config["training"]["epochs"] + 1):
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

            average_train_loss = total_loss / total_examples
            validation_accuracy = evaluate(
                model,
                validation_loader,
                device,
            )

            history.append({
                "epoch": epoch,
                "train_loss": average_train_loss,
                "validation_accuracy": validation_accuracy,
            })

            mlflow.log_metric(
                "train_loss",
                average_train_loss,
                step=epoch,
            )
            mlflow.log_metric(
                "validation_accuracy",
                validation_accuracy,
                step=epoch,
            )

            print(
                f"Epoch {epoch}/{config['training']['epochs']} "
                f"train_loss={average_train_loss:.4f} "
                f"validation_accuracy={validation_accuracy:.4f}"
            )

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_accuracy": validation_accuracy,
                        "config": config,
                    },
                    checkpoint_path,
                )

        with metrics_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "validation_accuracy",
                ],
            )
            writer.writeheader()
            writer.writerows(history)

        best_checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )
        model.load_state_dict(best_checkpoint["model_state_dict"])

        test_accuracy = evaluate(model, test_loader, device)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.log_artifact(
            str(checkpoint_path),
            artifact_path="checkpoints",
        )
        mlflow.log_artifact(str(metrics_path), artifact_path="results")

        print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
        print(f"Test accuracy using best checkpoint: {test_accuracy:.4f}")
        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()