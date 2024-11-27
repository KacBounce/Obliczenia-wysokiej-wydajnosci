import numpy as np
import time
import psutil

def classic_matrix_multiply(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


if __name__ == "__main__":
    sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    for N in sizes:
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        
        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss
        C = classic_matrix_multiply(A, B)
        
        mem_after = psutil.Process().memory_info().rss
        end_time = time.time()
        
        memory_usage = abs(mem_after - mem_before) / 1024

        print(f"Sekwencyjne mnozenie macierzy {N}x{
            N} zakonczone w czasie: {end_time - start_time:.6f} sekund\n
            Zuzycie pamieci : {memory_usage} KB")
