import matplotlib.pyplot as plt

# Load results
num_processes = [1, 4, 16, 25, 36, 64]  # Add more process counts if needed
results = {}

# Load data for each process count
for p in num_processes:
    with open(f"results_{p}.txt", "r") as f:
        lines = f.readlines()[1:]  # Skip header
        results[p] = []
        for line in lines:
            matrix_size, num_procs, sequential_time, parallel_time = map(
                float, line.strip().split(","))
            results[p].append((int(matrix_size), int(num_procs),
                               sequential_time, parallel_time))

# Extract matrix sizes (sorted and unique)
matrix_sizes = sorted(
    set(res[0] for res_list in results.values() for res in res_list))

# Ensure the speedup and efficiency lists have the same length as matrix_sizes
speedups = {p: [] for p in num_processes}
efficiencies = {p: [] for p in num_processes}

# Compute speedup and efficiency for each process count and matrix size
for p in num_processes:
    for matrix_size in matrix_sizes:
        # Filter the relevant data for the current matrix_size
        relevant_data = [entry for entry in results[p]
                         if entry[0] == matrix_size]
        if relevant_data:
            seq_time = relevant_data[0][2]
            par_time = relevant_data[0][3]
            speedups[p].append(seq_time / par_time)
            # Efficiency = Speedup / P
            efficiencies[p].append(speedups[p][-1] / p)

# Plot Speedup vs Matrix Size (logarithmic scale)
plt.figure(figsize=(10, 6))
for p in num_processes:
    plt.plot(matrix_sizes, speedups[p], label=f"S(p) for {
             p} processes", marker='o')
plt.xlabel("Matrix Size (N)")
plt.ylabel("Speedup (S(p))")
plt.title("Speedup vs Matrix Size (Logarithmic Scale)")
#plt.xscale('log')  # Logarithmic scale for the x-axis
plt.yscale('log')  # Logarithmic scale for the y-axis (if needed)
# Show grid for both major and minor ticks
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# Plot Efficiency vs Matrix Size (logarithmic scale)
plt.figure(figsize=(10, 6))
for p in num_processes:
    plt.plot(matrix_sizes, efficiencies[p], label=f"E(p) for {
             p} processes", marker='o')
plt.xlabel("Matrix Size (N)")
plt.ylabel("Efficiency (E(p))")
plt.title("Efficiency vs Matrix Size (Logarithmic Scale)")
#plt.xscale('log')  # Logarithmic scale for the x-axis
plt.yscale('log')  # Logarithmic scale for the y-axis (if needed)
# Show grid for both major and minor ticks
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# Plot Speedup vs Number of Processes (S(p) vs P)
plt.figure(figsize=(10, 6))
for matrix_size in matrix_sizes:
    # Get the speedup for the current matrix size
    matrix_speedups = [speedups[p][matrix_sizes.index(
        matrix_size)] for p in num_processes]
    plt.plot(num_processes, matrix_speedups, label=f"S(p) for {
             matrix_size}x{matrix_size} matrix", marker='o')
plt.xlabel("Number of Processes (P)")
plt.ylabel("Speedup (S(p))")
plt.title("Speedup vs Number of Processes (P)")
plt.axvline(x=24, color='red', linestyle='--', label="logical processors : 24")
plt.axvline(x=16, color='blue', linestyle='--', label="cores : 16")
#plt.xscale('log')  # Logarithmic scale for the x-axis
plt.yscale('log')  # Logarithmic scale for the y-axis (if needed)
plt.grid(True)
plt.legend()
plt.show()


# Plot Efficiency vs Number of Processes (E(p) vs P)
plt.figure(figsize=(10, 6))
for matrix_size in matrix_sizes:
    matrix_efficiencies = [efficiencies[p][matrix_sizes.index(
        # Get the efficiency for the current matrix size
        matrix_size)] for p in num_processes]
    plt.plot(num_processes, matrix_efficiencies, label=f"E(p) for {
             matrix_size}x{matrix_size} matrix", marker='o')
plt.xlabel("Number of Processes (P)")
plt.ylabel("Efficiency (E(p))")
plt.title("Efficiency vs Number of Processes (P)")
plt.axvline(x=24, color='red', linestyle='--', label="logical processors : 24")
plt.axvline(x=16, color='blue', linestyle='--', label="cores : 16")

#plt.xscale('log')  # Logarithmic scale for the x-axis
plt.yscale('log')  # Logarithmic scale for the y-axis (if needed)
plt.grid(True)
plt.legend()
plt.show()
