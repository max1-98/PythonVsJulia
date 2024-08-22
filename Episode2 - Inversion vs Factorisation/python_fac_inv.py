import numpy as np
import time
import pandas as pd

def benchmark_matrix_solvers(sizes,n):
    results = {
        'Matrix Size': [],
        'Inversion Time (s)': [],
        'Factorization Time (s)': []
    }
    for size in sizes:
        inversion_time = 0
        factorization_time = 0
        for i in range(n):
            # Create matrix and vector
            matrix = np.random.randint(-1000, 1001, size=(size, size))
            vector1 = np.random.randint(-1000, 1001, size=(size, 1))

            # Benchmark inversion
            start_time = time.perf_counter()
            m = np.linalg.inv(matrix)
            m @ vector1
            inversion_time += time.perf_counter() - start_time

            # Benchmark factorization (solving linear system)
            start_time = time.perf_counter()
            np.linalg.solve(matrix, vector1)
            factorization_time += time.perf_counter() - start_time

        # Store results
        results['Matrix Size'].append(size)
        results['Inversion Time (s)'].append(inversion_time/n)
        results['Factorization Time (s)'].append(factorization_time/n)

    return pd.DataFrame(results)

# Matrix sizes to benchmark
sizes = [2, 3, 5, 10, 30, 100, 1000]

# Run the benchmark and display results
results_df = benchmark_matrix_solvers(sizes,100)
print(results_df.to_string())

results_df.to_csv("python_benchmark_results2.csv", index=False)