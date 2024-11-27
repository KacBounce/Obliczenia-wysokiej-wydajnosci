from mpi4py import MPI  # https://github.com/mpi4py/mpi4py
import numpy as np
import time


def classic_matrix_multiply(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def compare_results(seq_result, mpi_result, tol=1e-6):
    if np.allclose(np.round(seq_result, decimals=4), np.round(mpi_result, decimals=4), atol=tol):
        print("Results are the same!")
    else:
        print("Results are different!")
        diff = np.abs(seq_result - mpi_result)
        print(f"Max difference: {np.max(diff)}")
        print(f"Mean difference: {np.mean(diff)}")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 16
grid_size = int(np.sqrt(size))
assert grid_size**2 == size, "Number of processes must be a square!"
assert N % grid_size == 0, "Matrix size N must be divisible by grid_size."
block_size = N // grid_size 

if __name__ == "__main__":
    if rank == 0:
        A = np.random.rand(N, N).astype(np.float64)
        B = np.random.rand(N, N).astype(np.float64)

        start_seq = time.time()
        seq_result = classic_matrix_multiply(A, B)
        
        seq_result = np.round(seq_result, decimals=4)
        end_seq = time.time()
        print(f"Sequential multiplication finished in: {
              end_seq - start_seq:.4f} seconds")
        
        print(f"Sequenrtial:\n{seq_result}")

        A_blocks = A.reshape(grid_size, block_size, grid_size, block_size).swapaxes(
            1, 2).reshape(grid_size**2, block_size, block_size)
        B_blocks = B.reshape(grid_size, block_size, grid_size, block_size).swapaxes(
            1, 2).reshape(grid_size**2, block_size, block_size)
    else:
        A_blocks = None
        B_blocks = None

    local_A = np.empty((block_size, block_size), dtype=np.float64)
    local_B = np.empty((block_size, block_size), dtype=np.float64)

    comm.Scatter(A_blocks, local_A, root=0)
    comm.Scatter(B_blocks, local_B, root=0)

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


    start_mpi = MPI.Wtime()
    for _ in range(grid_size):

        local_C += classic_matrix_multiply(local_A, local_B)
        local_C = np.round(local_C, decimals=4) 

        cart_comm.Sendrecv_replace(local_A, dest=left, source=right)
        cart_comm.Sendrecv_replace(local_B, dest=up, source=down)
    end_mpi = MPI.Wtime()

    if rank == 0:
        C_blocks = np.empty(
            (grid_size**2, block_size, block_size), dtype=np.float64)
    else:
        C_blocks = None

    comm.Gather(local_C, C_blocks, root=0)

    if rank == 0:
        C = np.zeros((N, N), dtype=np.float64)
        for i in range(grid_size):
            for j in range(grid_size):
                C[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                    C_blocks[i*grid_size + j]

        C = np.round(C, decimals=4) 
        
        print(f"Distributed:\n{np.round(seq_result, decimals=4)}")
        print(f"Distributed multiplication finished in: {
              end_mpi - start_mpi:.6f} seconds")
        compare_results(seq_result, C)
        print("Sequential checksum:", np.sum(seq_result))
        print("MPI checksum:", np.sum(C))
        print(np.round(C, decimals=4) == np.round(seq_result, decimals=4))
