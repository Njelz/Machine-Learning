import csv
import statistics
import time
from pathlib import Path

import torch
from thop import profile

from models import MLP, CNN, AttentionModel


def median_latency_ms(model, batch, repeats):
    # A few untimed runs let the model/runtime warm up.
    with torch.inference_mode():
        for _ in range(5):
            model(batch)

        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(batch)
            times.append(time.perf_counter() - start)

    return statistics.median(times) * 1000


def main():
    # Your installed PyTorch is CPU-only, so record CPU timings.
    device = torch.device("cpu")
    print(f"Device: {device}; PyTorch CPU threads: {torch.get_num_threads()}")

    models = {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": AttentionModel(),
    }

    rows = []

    for name, model in models.items():
        model.to(device)
        model.eval()

        # THOP profiling uses a one-image input.
        sample = torch.randn(1, 1, 28, 28, device=device)
        try:
            macs, thop_params = profile(
                model,
                inputs=(sample,),
                verbose=False,
            )
        except Exception as error:
            macs = None
            thop_params = None
            print(f"{name}: THOP could not profile this model: {error}")

        parameter_count = sum(p.numel() for p in model.parameters())

        batch1 = torch.randn(1, 1, 28, 28, device=device)
        batch128 = torch.randn(128, 1, 28, 28, device=device)

        latency_batch1 = median_latency_ms(model, batch1, repeats=100)
        latency_batch128 = median_latency_ms(model, batch128, repeats=20)

        print(f"\n{name}")
        print(f"  Parameters (PyTorch): {parameter_count:,}")
        if macs is not None:
            print(f"  THOP MACs for batch 1: {macs:,.0f}")
            print(f"  THOP parameters: {thop_params:,.0f}")
        print(f"  Batch 1 median latency: {latency_batch1:.4f} ms")
        print(f"  Batch 128 median latency: {latency_batch128:.4f} ms")

        rows.append([
            name,
            parameter_count,
            "" if macs is None else int(macs),
            f"{latency_batch1:.6f}",
            f"{latency_batch128:.6f}",
        ])

    output = Path(__file__).parent.parent / "results" / "cost_measurements.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "parameters",
            "thop_macs_batch1",
            "median_latency_batch1_ms",
            "median_latency_batch128_ms",
        ])
        writer.writerows(rows)

    print(f"\nSaved measurements to: {output}")


if __name__ == "__main__":
    main()
