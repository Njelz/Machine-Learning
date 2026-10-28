from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    # Download and load Fashion-MNIST's official training split (60,000).
    full_train = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Use 55,000 examples for training and 5,000 for validation.
    train_set, validation_set = random_split(
        full_train,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(42),
    )

    # Load Fashion-MNIST's separate official test split (10,000).
    test_set = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    validation_loader = DataLoader(
        validation_set, batch_size=128, shuffle=False
    )
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    images, labels = next(iter(train_loader))

    print(f"Training examples: {len(train_set)}")
    print(f"Validation examples: {len(validation_set)}")
    print(f"Test examples: {len(test_set)}")
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"First batch labels: {labels[:10].tolist()}")


if __name__ == "__main__":
    main()
