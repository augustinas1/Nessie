# Run SSA simulations to obtain histograms for the parameters inferred
# using optimize.jl, saving the result in sims.jld2
using JLD2
using Catalyst
using Random

include("../generate_data.jl")

include("network.jl")

@load joinpath(DATA_DIR, "params.jld2") estimates

sims = []
for logps in estimates
    ps = 10 .^ logps
    yy_sim = solver(tt, ps, 1e6)[1]                          
    push!(sims, yy_sim)                                                         

    @save joinpath(DATA_DIR, "sims.jld2") sims
end 
