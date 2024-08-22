using CSV, DataFrames, LinearAlgebra, BenchmarkTools, Random

function benchmark_matrix_solvers(nums)
    results = DataFrame(matrix_size = Int[], inversion_time = Float64[], factorization_time = Float64[])
    
    for n in nums
        # Create matrix and vector outside the benchmark loop
        matrix = rand(-1000:1000, n, n)
        vector = rand(-1000:1000, n, 1)

        # Benchmark inversion
        inversion_time = @belapsed inv($matrix) * $vector
        
        # Benchmark factorization
        factorization_time = @belapsed $matrix \ $vector

        # Append results to DataFrame
        push!(results, [n, inversion_time, factorization_time])
    end
    
    return results
end

nums = [2, 3, 5, 10, 30, 100, 1000]
results_df = benchmark_matrix_solvers(nums)

# Print results
println(results_df)

# You can save the results to a CSV file if needed
CSV.write("julia_benchmark_results.csv", results_df)

