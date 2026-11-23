For framework inspection, I traced a small 2→8→1 PyTorch model with torch.jit.trace. After warm-up, eager and traced outputs matched exactly. Median call times were 0.08004 ms for eager execution and 0.04257 ms for traced execution. In this benchmark, tracing was about 1.9 times faster, though the result applies only to this small model and test setup.

I also compared matrix multiplication using a non-contiguous transposed view and a contiguous copy. Their median times were 1.415 ms and 1.336 ms, respectively, so the non-contiguous view was about 5.9% slower in this test. This shows that layout can affect speed, but the difference depends on the operation and hardware.

Autodiff engine and gradient validation
I implemented a scalar Value node with addition, subtraction, multiplication, powers, ReLU, sigmoid, logarithm, and binary cross-entropy operations. The backward pass uses a topological ordering and accumulates gradients when a value is reused.

For ((a*b + c)**2).relu(), my engine produced an output of 49.0 and gradients of 28.0, 42.0, and 14.0. These matched PyTorch. The maximum absolute difference from the numerical gradients was 8.97 × 10⁻¹⁰, below the required 10⁻⁶.

make_moons model and timing
I trained a 2→8→1 network with sigmoid output and binary cross-entropy on 500 make_moons examples (400 for training and 100 for testing). I used 100 epochs and SGD with a learning rate of 0.1. Across five runs, PyTorch's median training time was 0.1248 seconds and my custom autodiff engine's median was 37.9827 seconds. PyTorch test accuracy ranged from 81% to 84%; custom-model test accuracy ranged from 82% to 86%. The custom engine was much slower in this experiment, while test accuracy was similar.

Computer and timing details
The make_moons times are medians of five 100-epoch runs. Framework timings used 20 warm-up calls followed by five rounds of 100 calls.

The training tensor had shape (400, 2), dtype float32, and was contiguous. Its transposed view had shape (2, 400), dtype float32, and was non-contiguous. Tracing may help when a stable model is called repeatedly by reducing some execution overhead, but the benefit depends on the model, hardware, and workload.
