# Lab 3: Architectures and Their Costs

## Method

I used Fashion-MNIST with `ToTensor()` and normalization using mean 0.2860 and standard deviation 0.3530. I split the official 60,000-example training set into 55,000 training examples and 5,000 validation examples, and used the separate official 10,000-example test set. The batch size was 128.

I implemented the specified MLP, CNN, and two-layer Transformer attention model in PyTorch. Each model was trained for five epochs using Adam with a learning rate of 0.001. Inference timings were measured on CPU after warm-up, using the median of 100 runs for batch size 1 and 20 runs for batch size 128.

## Parameter Counts and Test Accuracy

| Model | Parameters | Test accuracy |
|---|---:|---:|
| MLP | 235,146 | 87.79% |
| CNN | 20,490 | 89.17% |
| Attention | 70,794 | 84.34% |

The manual asks for models with roughly 100,000–150,000 parameters, but the listed layer definitions produce counts outside that range. I followed the specified layer definitions and reported their actual parameter counts.

## Inference Latency

| Model | Batch 1 median (ms) | Batch 128 median (ms) |
|---|---:|---:|
| MLP | 0.3236 | 1.8950 |
| CNN | 0.9646 | 25.6000 |
| Attention | 2.5267 | 18.5901 |

The measurements were taken on the CPU using PyTorch with four CPU threads. They are specific to this computer and software setup.

## Hand FLOP Calculations and THOP

For one representative layer of each model:

- **MLP, Linear 784 → 256:**  
  `2 × 784 × 256 = 401,408 FLOPs`
- **CNN, first 3×3 convolution (1 → 16 channels, output 28×28):**  
  `2 × 3 × 3 × 1 × 16 × 28 × 28 = 225,792 FLOPs`
- **Attention, QKV projections (sequence length 16, d_model 64):**  
  `2 × 16 × 64 × 3 × 64 = 393,216 FLOPs`

THOP reported MACs (multiply-accumulate operations) for a batch of one: 234,752 for the MLP, 1,031,744 for the CNN, and 50,816 for the attention model. A MAC is often treated as approximately two FLOPs, so these numbers are not directly the same unit as the hand-calculated FLOPs.

THOP reported 3,850 parameters for the attention model, while PyTorch counted 70,794. THOP did not account for the Transformer encoder layers in its report, so its attention-model operation and parameter counts are incomplete. I use PyTorch’s count for the model’s parameter total.

The measurements are saved in `results/cost_measurements.csv`, and the trade-off plots are saved in `results/tradeoffs.png`.

## Trade-offs and Access Patterns

The CNN had the highest test accuracy (89.17%) and the fewest parameters (20,490), but it was slower than the MLP in these CPU measurements. Convolutions reuse the same small filters over local image regions, which makes use of spatial structure and shares weights. However, the measured latency also depends on how efficiently the software and CPU implement convolution.

The MLP had the lowest batch-1 latency and reached 87.79% test accuracy, but it had the most parameters. It flattens the image and uses dense layers, so it does not explicitly preserve local pixel neighborhoods.

The attention model had 70,794 parameters and 84.34% test accuracy. It divides the image into patches and applies Transformer layers across the patch sequence. Its batch-1 latency was highest in this run. Attention can relate information across patches, but its additional operations and implementation overhead can affect latency.

Overall, the CNN gave the best accuracy with the fewest parameters in this experiment. The MLP was fastest for one-image inference. These results show that parameter count alone does not determine latency or accuracy; data access patterns, operator implementation, batch size, and hardware also matter.
