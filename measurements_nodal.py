from mpi4py import MPI  # https://github.com/mpi4py/mpi4py
import numpy as np
import time
import matplotlib.pyplot as plt
import psutil


def classic_matrix_multiply(A, B):
    """Performs classic matrix multiplication."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def compare_results(seq_result, mpi_result, tol=1e-6):
    """Compares sequential and MPI results."""
    if np.allclose(seq_result, mpi_result, atol=tol):
        print("Results are the same!")
    else:
        print("Results are different!")
        diff = np.abs(seq_result - mpi_result)
        print(f"Max difference: {np.max(diff)}")
        print(f"Mean difference: {np.mean(diff)}")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

input_sizes = [64, 128, 256, 512, 1024]  
block_timings = {"Scatter": [], "Compute": [], "Gather": [], "Total": []}
memory_usages = []
all_timings = []


grid_size = int(np.sqrt(size))
assert grid_size**2 == size, "Number of processes must be a perfect square!"
block_size = None  

if __name__ == "__main__":
    for N in input_sizes:
        assert N % grid_size == 0, "Matrix size N must be divisible by grid_size!"
        block_size = N // grid_size

        if rank == 0:
            A = np.random.rand(N, N).astype(np.float64)
            B = np.random.rand(N, N).astype(np.float64)

            A_blocks = A.reshape(grid_size, block_size, grid_size, block_size).swapaxes(
                1, 2).reshape(grid_size**2, block_size, block_size)
            B_blocks = B.reshape(grid_size, block_size, grid_size, block_size).swapaxes(
                1, 2).reshape(grid_size**2, block_size, block_size)
        else:
            A_blocks = None
            B_blocks = None

        local_A = np.empty((block_size, block_size), dtype=np.float64)
        local_B = np.empty((block_size, block_size), dtype=np.float64)

        start_scatter = MPI.Wtime()
        mem_before_scatter = psutil.Process().memory_info().rss
        comm.Scatter(A_blocks, local_A, root=0)
        comm.Scatter(B_blocks, local_B, root=0)
        mem_after_scatter = psutil.Process().memory_info().rss
        end_scatter = MPI.Wtime()

        scatter_memory_usage = (
            abs(mem_after_scatter - mem_before_scatter) / (1024))


        dims = [grid_size, grid_size]
        periods = [True, True] 
        cart_comm = comm.Create_cart(dims, periods=periods, reorder=True)

        coords = cart_comm.Get_coords(rank)
        row_rank, col_rank = coords
        left, right = cart_comm.Shift(1, -1)
        up, down = cart_comm.Shift(0, -1)

        for _ in range(row_rank):
            cart_comm.Sendrecv_replace(local_A, dest=left, source=right)
        for _ in range(col_rank):
            cart_comm.Sendrecv_replace(local_B, dest=up, source=down)

        local_C = np.zeros((block_size, block_size), dtype=np.float64)

        start_compute = MPI.Wtime()
        mem_before_compute = psutil.Process().memory_info().rss
        for _ in range(grid_size):
            local_C += classic_matrix_multiply(local_A, local_B)
            cart_comm.Sendrecv_replace(local_A, dest=left, source=right)
            cart_comm.Sendrecv_replace(local_B, dest=up, source=down)
        mem_after_compute = psutil.Process().memory_info().rss
        end_compute = MPI.Wtime()

        compute_memory_usage = (
            abs(mem_after_compute - mem_before_compute)) / (1024)

        start_gather = MPI.Wtime()
        mem_before_gather = psutil.Process().memory_info().rss
        if rank == 0:
            C_blocks = np.empty(
                (grid_size**2, block_size, block_size), dtype=np.float64)
        else:
            C_blocks = None

        comm.Gather(local_C, C_blocks, root=0)
        mem_after_gather = psutil.Process().memory_info().rss
        end_gather = MPI.Wtime()

        gather_memory_usage = (
            abs(mem_after_gather - mem_before_gather) / (1024))  

        block_timings["Scatter"].append(end_scatter - start_scatter)
        block_timings["Compute"].append(end_compute - start_compute)
        block_timings["Gather"].append(end_gather - start_gather)
        block_timings["Total"].append(
            (end_gather - start_scatter)) 

        memory_usages.append({
            "Scatter": scatter_memory_usage,
            "Compute": compute_memory_usage,
            "Gather": gather_memory_usage,
            "Total": scatter_memory_usage + compute_memory_usage + gather_memory_usage
        })

        if rank == 0:
            C = np.zeros((N, N), dtype=np.float64)
            for i in range(grid_size):
                for j in range(grid_size):
                    C[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                        C_blocks[i*grid_size + j]

            start_seq = time.time()
            mem_before = psutil.Process().memory_info().rss
            seq_result = classic_matrix_multiply(A, B)

            mem_after = psutil.Process().memory_info().rss
            end_seq = time.time()
            
            
            
            seq_time = end_seq - start_seq
            seq_memory_usage = abs(mem_after - mem_before) / 1024
            
            compare_results(seq_result, C)

            print(f"\nMatrix Size: {N}x{N}")
            print(f"Scatter Time: {
                  block_timings['Scatter'][-1]:.6f} s, Memory Usage: {scatter_memory_usage:.2f} KB")
            print(f"Compute Time: {
                  block_timings['Compute'][-1]:.6f} s, Memory Usage: {compute_memory_usage:.2f} KB")
            print(f"Gather Time: {
                  block_timings['Gather'][-1]:.6f} s, Memory Usage: {gather_memory_usage:.2f} KB")
            print(f"Total Time: {block_timings['Total'][-1]:.6f} s, Total Memory Usage: {
                  scatter_memory_usage + compute_memory_usage + gather_memory_usage} KB")
            print(f"Total Squential Time: {
                  seq_time:.6f} s, Sequential memory usage : {seq_memory_usage} KB")

    if rank == 0:
        plt.figure(figsize=(10, 6))
        for key, timings in block_timings.items():
            plt.plot(input_sizes, timings, label=key, marker='o')
        plt.legend()
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("Time (s)")
        plt.title("Timing Breakdown by Phase")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        for key in memory_usages[0].keys():
            memory = [m[key]
                      for m in memory_usages]
            plt.plot(input_sizes, memory, label=f"{
                     key} Memory Usage", marker='o')
        plt.legend()
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("Memory Usage (KB)")
        plt.title("Memory Usage by Phase")
        plt.grid(True)
        plt.show()

        communication_costs = [block_timings["Scatter"][i] +
                               block_timings["Gather"][i] for i in range(len(input_sizes))]
        plt.figure(figsize=(10, 6))
        plt.plot(input_sizes, communication_costs,
                 label="Communication Cost", marker='o')
        plt.xlabel("Matrix Size (N)")
        plt.ylabel("Time (s)")
        plt.title("Communication Cost (Scatter + Gather)")
        plt.grid(True)
        plt.legend()
        plt.show()
