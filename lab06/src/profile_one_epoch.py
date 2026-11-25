import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lab03" / "src"))
from models import CNN


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
    train_set, _ = random_split(
        full_train_set,
        [len(full_train_set) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config["dataset"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        generator=torch.Generator().manual_seed(seed),
    )

    model = CNN().to(device)
    checkpoint_path = (
        REPO_ROOT
        / config["checkpoint"]["directory"]
        / config["checkpoint"]["filename"]
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    loss_function = nn.CrossEntropyLoss()

    iterator = iter(train_loader)
    phase_names = [
        "data_loading",
        "forward_pass",
        "backward_pass",
        "optimizer_step",
    ]

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        for _ in range(len(train_loader)):
            with record_function("data_loading"):
                images, labels = next(iterator)
                images = images.to(device)
                labels = labels.to(device)

            with record_function("forward_pass"):
                logits = model(images)
                loss = loss_function(logits, labels)

            with record_function("backward_pass"):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

    # Sum the self CPU time of the four labelled phases.
    phase_times_ms = {name: 0.0 for name in phase_names}

    for event in prof.key_averages():
        if event.key in phase_times_ms:
            phase_times_ms[event.key] = event.cpu_time_total / 1000

    total_ms = sum(phase_times_ms.values())

    results_dir = REPO_ROOT / "lab06" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_csv = results_dir / "profile_breakdown.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["phase", "time_ms", "share_percent"])

        for name in phase_names:
            time_ms = phase_times_ms[name]
            share = 100 * time_ms / total_ms if total_ms else 0
            writer.writerow([name, f"{time_ms:.3f}", f"{share:.2f}"])
            print(f"{name}: {time_ms:.3f} ms ({share:.2f}%)")

    trace_path = results_dir / "profile_trace.json"
    prof.export_chrome_trace(str(trace_path))

    print(f"\nSaved breakdown to: {output_csv}")
    print(f"Saved profiler trace to: {trace_path}")


if __name__ == "__main__":
    main()