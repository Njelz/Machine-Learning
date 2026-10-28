import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    results_dir = Path(__file__).parent.parent / "results"

    accuracies = {
        row["model"]: float(row["test_accuracy"])
        for row in read_csv(results_dir / "test_accuracy.csv")
    }
    costs = {
        row["model"]: row
        for row in read_csv(results_dir / "cost_measurements.csv")
    }

    models = ["MLP", "CNN", "Attention"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model in models:
        parameters = int(costs[model]["parameters"])
        batch1_latency = float(
            costs[model]["median_latency_batch1_ms"]
        )
        accuracy_percent = accuracies[model] * 100

        axes[0].scatter(parameters, accuracy_percent, s=70)
        axes[0].annotate(
            model, (parameters, accuracy_percent),
            xytext=(5, 5), textcoords="offset points"
        )

        axes[1].scatter(batch1_latency, accuracy_percent, s=70)
        axes[1].annotate(
            model, (batch1_latency, accuracy_percent),
            xytext=(5, 5), textcoords="offset points"
        )

    axes[0].set_xlabel("Number of parameters")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Accuracy vs. model size")

    axes[1].set_xlabel("Batch-1 median latency (ms)")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Accuracy vs. inference latency")

    for axis in axes:
        axis.grid(True, alpha=0.3)

    fig.suptitle("Fashion-MNIST architecture trade-offs")
    fig.tight_layout()

    output = results_dir / "tradeoffs.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)

    print(f"Saved plots to: {output}")


if __name__ == "__main__":
    main()
