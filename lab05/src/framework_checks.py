import statistics
import time

import torch
import torch.nn as nn


def median_time_ms(function, warmup=20, repeats=5, iterations=100):
    """Return median time per call, in milliseconds."""
    with torch.inference_mode():
        for _ in range(warmup):
            function()

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()

            for _ in range(iterations):
                function()

            elapsed = time.perf_counter() - start
            timings.append((elapsed / iterations) * 1000)

    return statistics.median(timings)


def main():
    torch.set_num_threads(1)

    # A small 2 -> 8 -> 1 model, like the make_moons model.
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    ).eval()

    example_input = torch.randn(32, 2)

    # Trace the model and display its computation graph.
    traced_model = torch.jit.trace(model, example_input)

    print("Traced model graph:")
    print(traced_model.graph)

    # Check that eager and traced models produce matching outputs.
    with torch.inference_mode():
        eager_output = model(example_input)
        traced_output = traced_model(example_input)
        max_difference = (eager_output - traced_output).abs().max().item()

    print(f"\nMaximum eager/traced output difference: {max_difference:.8f}")

    # Warmed-up timing comparison.
    eager_ms = median_time_ms(lambda: model(example_input))
    traced_ms = median_time_ms(lambda: traced_model(example_input))

    print(f"Eager median time:  {eager_ms:.6f} ms per call")
    print(f"Traced median time: {traced_ms:.6f} ms per call")

    # Compare an operation on a non-contiguous view and a contiguous copy.
    base = torch.randn(512, 512)
    non_contiguous = base.T
    contiguous = non_contiguous.contiguous()
    right_hand_matrix = torch.randn(512, 256)

    print("\nMatrix input checks:")
    print("Non-contiguous shape:", non_contiguous.shape)
    print("Non-contiguous:", non_contiguous.is_contiguous())
    print("Contiguous copy:", contiguous.is_contiguous())

    noncontiguous_ms = median_time_ms(
        lambda: non_contiguous @ right_hand_matrix
    )
    contiguous_ms = median_time_ms(
        lambda: contiguous @ right_hand_matrix
    )

    print(f"Non-contiguous median time: {noncontiguous_ms:.6f} ms")
    print(f"Contiguous median time:     {contiguous_ms:.6f} ms")


if __name__ == "__main__":
    main()