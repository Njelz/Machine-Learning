# Lab 1: Environment and First System Measurements

## Goal

The goal of this lab was to set up the Python environment, train two baseline classifiers on the Breast Cancer Wisconsin dataset, and compare their accuracy, training time, inference time, saved model size, and memory use.

## Method

I used Python 3.11.8 and the built-in Breast Cancer Wisconsin dataset from scikit-learn. The dataset has 569 examples and 30 features. I split it into 70% training data and 30% test data, using stratification and random state 42.

I trained logistic regression with standardized features and a random forest with 100 trees. I measured accuracy on the held-out test data. For training time, I ran one warm-up and five timed fits, then recorded the median. For inference time, I warmed up each model and timed 100 predictions of one test example, then recorded the median. I saved each trained model with joblib to measure its file size.

Memory was measured with `memory_profiler` and `psutil`. These numbers are peak memory for the Python process, not just the model, so they are approximate and include the Python and library overhead.

## Results

| Model | Test accuracy | Median training time | Median single-example inference | Saved model size | Peak process memory during training |
|---|---:|---:|---:|---:|---:|
| Logistic regression (standardized) | 0.9883 | 0.041196 s | 0.756000 ms | 2.23 KB | 105.23 MiB |
| Random forest | 0.9357 | 0.325653 s | 9.251850 ms | 284.07 KB | 105.55 MiB |

Peak process memory during inference was 105.24 MiB for logistic regression and 105.68 MiB for the random forest. The measured increase over the starting process memory was small, so these memory figures should be treated as rough process-level measurements.

## Conclusions

1. Logistic regression had higher test accuracy than the random forest on this split. Its score was 0.9883 compared with 0.9357.

2. Logistic regression was faster to train and faster to make a single prediction in my measurements. The random forest also produced a much larger saved model file.

3. Based on the manual’s latency and file-size budgets, logistic regression looks suitable for Cloud, Edge, or Mobile use. Its saved file is also below the TinyML file-size limit, but I would need a better measurement of actual device memory use before deciding whether it is suitable for TinyML. The random forest file is over the TinyML size limit. The memory measurements here are process-level estimates, not direct device measurements.