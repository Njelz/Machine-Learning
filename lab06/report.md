## Task 1: Reproducible training pipeline

I created a YAML-driven pipeline using the Lab 3 Fashion-MNIST CNN. The configuration records the random seed, data settings, model, optimizer, learning rate, and number of epochs. The pipeline saves the checkpoint with the best validation accuracy and logs parameters, metrics, and artifacts to MLflow.

After five epochs, the best validation accuracy was 89.60% and test accuracy was 89.17%. The checkpoint, training metrics, and configuration were saved as artifacts.


## Task 2: Profiler breakdown

I profiled one training epoch on CPU using batch size 128 and `num_workers=0`. The profiler attributed the largest share of time to data loading.

| Phase | Share of profiled CPU time |
|---|---:|
| Data loading | 66.30% |
| Forward pass | 15.62% |
| Backward pass | 15.79% |
| Optimizer step | 2.29% |


## Task 3: Data-loader worker study

I compared Fashion-MNIST data-loading throughput using a batch size of 128. For each worker setting, I used one warm-up pass and took the median of three measured passes.

| `num_workers` | Median pass time (s) | Throughput (samples/s) |
|---:|---:|---:|
| 0 | 14.74 | 3,732 |
| 1 | 20.77 | 2,647 |
| 2 | 10.89 | 5,049 |
| 4 | 6.03 | 9,123 |
| 8 | 4.95 | 11,102 |

In this data-loading-only benchmark, throughput was highest with 8 workers. One worker was slower than zero, showing that worker-process overhead can outweigh its benefit at low worker counts. The results and throughput plot are saved in `results/worker_benchmark.csv` and `results/worker_throughput.png`.


## Task 4: Batch-Size Study

| Batch size | Throughput (samples/s) | Peak process memory (MB) | Test accuracy after 5 epochs |
| ---------: | ---------------------: | -----------------------: | ---------------------------: |
|         32 |                  716.0 |                    389.4 |                       90.29% |
|         64 |                  831.1 |                    394.7 |                       90.08% |
|        128 |                  879.6 |                    413.1 |                       89.17% |
|        256 |                  696.4 |                    442.7 |                       88.81% |
|        512 |                  927.0 |                    477.3 |                       87.81% |


## Task 5: Optimizers and Learning-Rate Schedules

| Optimizer      | Scheduler         | Test accuracy |
| -------------- | ----------------- | ------------: |
| Adam           | StepLR            |        89.17% |
| Adam           | CosineAnnealingLR |        90.24% |
| Adam           | ReduceLROnPlateau |        89.83% |
| SGD + momentum | StepLR            |        84.96% |
| SGD + momentum | CosineAnnealingLR |        86.25% |
| SGD + momentum | ReduceLROnPlateau |        86.21% |

The comparison used a learning rate of 0.001. SGD used momentum 0.9. StepLR used `step_size=3` and `gamma=0.1`; CosineAnnealingLR used `T_max=10`; ReduceLROnPlateau used `factor=0.1` and `patience=2`.


## Task 6: Gradient accumulation

I compared direct batch-256 training with batch-64 training using gradient accumulation over four mini-batches. Both runs used the same model, data split, initialization, optimizer, and five-epoch budget.

The training and validation curves were very similar. Final test accuracy was 88.81% for direct batch 256 and 88.84% for batch 64 with accumulation. This indicates similar training behavior in this experiment. Results and curves are saved under `results/gradient_accumulation`.


## Task 7: Checkpointing and Resuming Training

To test checkpoint resumption, I trained the CNN through epoch 3, saved the model and optimizer state, then resumed training from the checkpoint. The resumed run continued at epoch 4 and completed epoch 5. Training loss decreased from 0.3071 at epoch 3 to 0.2633 at epoch 5, while validation accuracy increased from 88.84% to 89.86%. Final test accuracy was 89.38%. This showed that training continued successfully from the saved checkpoint.
