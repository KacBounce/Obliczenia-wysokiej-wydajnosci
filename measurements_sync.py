import numpy as np
import time
import psutil
import matplotlib.pyplot as plt


def classic_matrix_multiply(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def measure_sequential_multiplication(input_sizes):
    timings = []
    mem_usages = []

    for N in input_sizes:
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)


        start_mem = psutil.virtual_memory().used

        start_time = time.time()
        C = classic_matrix_multiply(A, B)
        end_time = time.time()
        
        end_mem = psutil.virtual_memory().used

        timings.append(end_time - start_time)
        mem_usages.append(end_mem - start_mem)

    return timings, mem_usages


if __name__ == "__main__":
    input_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  

    timings, mem_usages = measure_sequential_multiplication(input_sizes)

    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, timings, label="Execution Time (s)", marker='o')
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Time (s)")
    plt.title("Synchronous Matrix Multiplication Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, [m / (1024)
             for m in mem_usages], label="Memory Usage (MB)", marker='o')
    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage for Synchronous Calculations")
    plt.legend()
    plt.grid(True)
    plt.show()
