import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import MLP, CNN, AttentionModel


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


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")
    data_dir = Path(__file__).resolve().parents[1] / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    full_train = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    train_set, validation_set = random_split(
        full_train,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(42),
    )
    test_set = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    validation_loader = DataLoader(
        validation_set, batch_size=128, shuffle=False
    )
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model_classes = {
        "MLP": MLP,
        "CNN": CNN,
        "Attention": AttentionModel,
    }

    results = []

    for name, model_class in model_classes.items():
        # Reset the seed so each model has a repeatable initialization.
        torch.manual_seed(42)
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_function = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
            num_workers=0,
        )

        for epoch in range(1, 6):
            model.train()
            total_loss = 0.0
            seen = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                seen += labels.size(0)

            train_loss = total_loss / seen
            validation_accuracy = evaluate(
                model, validation_loader, device
            )

            print(
                f"{name}: epoch {epoch}/5, "
                f"train loss={train_loss:.4f}, "
                f"validation accuracy={validation_accuracy:.4f}"
            )

        test_accuracy = evaluate(model, test_loader, device)
        print(f"{name}: final test accuracy={test_accuracy:.4f}\n")

        results.append([name, f"{test_accuracy:.6f}"])

    output_file = (
        Path(__file__).parent.parent / "results" / "test_accuracy.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "test_accuracy"])
        writer.writerows(results)

    print(f"Saved test accuracies to: {output_file}")


if __name__ == "__main__":
    main()
