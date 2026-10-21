# Lab 2: A Neural Network from Scratch

## Goal

Implement a two-hidden-layer neural network using NumPy, check its forward pass and gradients, and compare learning rates, hidden widths, accuracy, and pass timing.

## Method

I used the scikit-learn digits dataset, divided pixel values by 16, one-hot encoded the labels, and made a stratified 70/30 train/test split with random state 42. I implemented a network with two ReLU hidden layers and a 10-class softmax output. I trained with mini-batches of 32 for 50 epochs, using widths 16 and 64 and learning rates 0.001, 0.01, and 0.1.

The forward-pass output shape and probability sums were checked. The untrained loss was about 2.3026. I also compared analytical and numerical gradients; the check reported 10/10 passing for each width. The numerical check used higher-precision loss calculations and avoided perturbations that crossed ReLU boundaries. Forward and backward times are medians of 100 runs on a batch of 32.

## Results

| Width | Learning rate | Final test accuracy |
|---:|---:|---:|
| 16 | 0.001 | 0.1019 |
| 16 | 0.01 | 0.1019 |
| 16 | 0.1 | 0.5037 |
| 64 | 0.001 | 0.1019 |
| 64 | 0.01 | 0.1019 |
| 64 | 0.1 | 0.9056 |

The loss curves are saved in `results/loss_curves.png`.

| Width | Forward median | Backward median |
|---:|---:|---:|
| 16 | 0.098700 ms | 0.116850 ms |
| 64 | 0.164500 ms | 0.214550 ms |

Timing results are saved in `results/pass_times.csv`; training logs and accuracies are saved in `results/training_log.csv` and `results/test_accuracy.csv`.

## Conclusions

1. The learning rate made a large difference in these runs. Rates 0.001 and 0.01 stayed close to chance accuracy, while 0.1 performed better.
2. Width 64 with learning rate 0.1 had the highest test accuracy in this experiment: 0.9056.
3. Backward passes took longer than forward passes for both widths, and both passes took longer at width 64. This is consistent with backpropagation doing additional matrix operations to calculate gradients.
