import time
from memory_profiler import profile
from micrograd.nn import MLP


@profile
def main():
    """The main training loop function."""
    print("Starting training loop...")
    n = MLP(3, [4, 4, 1])

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    num_iterations = 2000
    for k in range(num_iterations):

        # Forward pass
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # Backward pass
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # Update
        for p in n.parameters():
            p.data += -0.05 * p.grad

        # Print progress less frequently due to more iterations
        if k % 200 == 0:
            print(f"Iteration {k}, Loss: {loss.data:.4f}")

    print("Training loop finished.")


if __name__ == '__main__':
    start_time = time.perf_counter()

    main()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("\n--- Profiling Results ---")
    print(f"Total Execution Time: {elapsed_time:.4f} seconds")
    print("-------------------------")
    print("Memory usage report is printed above by the 'memory_profiler'.")