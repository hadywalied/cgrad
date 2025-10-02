# Benchmark: C++ vs. Python Performance

This report compares the performance of a simple neural network training loop implemented in both **C++** and **Python**. Both versions were run with the same workload (2,000 iterations) to ensure a fair comparison of execution speed and memory consumption.

## Summary of Results

The C++ implementation demonstrates significantly higher performance, running approximately **17.3 times faster** while consuming about **10 times less memory** for the main loop's operations.

| Language | Execution Time | Memory Consumed |
| :--- |:---------------| :--- |
| **C++** | 1.095 s        | 560 KB |
| **Python** | 18.926 s       | \~5,734 KB (5.6 MiB) |

-----

## Detailed Results

### C++ Performance

The C++ program was measured for total execution time and the increase in process memory usage before and after the training loop.

```
--- Profiling Results ---
Total execution time: 1095.08 ms
                       1.09508 s
-------------------------
Memory before loop: 4220 KB
Memory after loop:  4780 KB
Approx. memory consumed by loop: 560 KB
-------------------------
```

### Python Performance

The Python script was profiled using `time.perf_counter()` for timing and `memory-profiler` for a detailed, line-by-line memory analysis.

* **Total Execution Time**: 18.9258 seconds
* **Memory Usage**: The process memory grew from a baseline of **56.5 MiB** to a peak of **62.1 MiB** during the loop, indicating a consumption of approximately **5.6 MiB**.

The line-by-line profile shows that the list comprehension for the forward pass (`ypred = [n(x) for x in xs]`) was responsible for the single largest memory allocation increment during the loop.

```
Filename: C:\Users\hady\CLionProjects\cgrad\micrograd_bm\main.py
Line #    Mem usage    Increment   Line Contents
=============================================================
...
 10        56.5 MiB      0.0 MiB       print("Starting training loop...")
...
 22        62.1 MiB      0.0 MiB       for k in range(2000):
...
 25        62.1 MiB      5.1 MiB           ypred = [n(x) for x in xs]
...
 41        62.1 MiB      0.0 MiB       print("Training loop finished.")

--- Profiling Results ---
Total Execution Time: 18.9258 seconds
```