using CSV, DataFrames, LinearAlgebra, Plots, BenchmarkTools

mutable struct LinearModel
    step_size::Float64
    max_iter::Int64
    eps::Float64
    theta::Vector{Float64}
    verbose::Bool
end

function LinearModel(theta::Vector{Float64}=zeros(1),step_size::Float64=0.2, max_iter::Int64=100, eps::Float64=1e-5, verbose::Bool=true)

    return LinearModel(step_size,max_iter,eps,theta,verbose)

end

function sigm(x)
    return 1/(1+exp(-x))
end

function fit(model::LinearModel, x,y)

    m,n=size(x)
    model.theta = zeros(n)
    
    i = 1
    while true
        oldtheta = copy(model.theta)
        ht = sigm.(-x*model.theta)
        H = H = 1/m * x' * diagm(ht .* (1 .- ht)) * x
        gradJ = 1/m *x'*(y.-ht)
        model.theta -= H \ gradJ

        i += 1
        if norm(model.theta - oldtheta, 1) < model.eps
            break
        end 
    end
end

function plt(x,y)
    scatter!(x[:, 2], x[:, 3], zcolor=y)
end

function pltfit(model::LinearModel, x, y)
    x1 = minimum(x[:,2])
    x2 = maximum(x[:,2])
    plot!([x1,x2], [-(x1*model.theta[2]+model.theta[1])/model.theta[3],-(x2*model.theta[2]+model.theta[1])/model.theta[3]])
end

scatter()
model = LinearModel()

function csv_to_matrix(filename::String)
    # Read the CSV file into a DataFrame
    df = CSV.read(filename, DataFrame)
    n = size(df[1,:])[1]

    x = Matrix(hcat(ones(size(df, 1)), df[:, 1:end-1])) 
    y = Vector(df[:,n])
    return (x,y)
end

# Example usage:
filename = "file_name.csv" # Replace with your CSV filename
x,y = csv_to_matrix(filename)

plt(x,y)

# Only benchmarks the fit function
@btime fitLOGREG(model,x,y)

pltfit(model,x,y)
savefig("my_plot.png") 


