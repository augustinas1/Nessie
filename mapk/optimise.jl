# Use Nessie to find parameters minimising the Hellinger distance to the 
# experimental data, saving the results in params.jld2
using JLD2
using Catalyst
using Random
using BlackBoxOptim

include("../train_NN.jl")
include("../generate_data.jl")

include("network.jl")
include("inference.jl")

@load joinpath(DATA_DIR, "model.jld2") model_params
model = build_model(6)
Flux.loadparams!(model, model_params) 

estimates = []

p = Progress(100)
Threads.@threads for i in 1:100
    opt_result = bboptimize(p -> loss_hellinger_mapk(p, model, tt, yobs); SearchRange = [ tuple(logranges[i,:]...) for i in 1:d ], TraceMode=:silent)
    push!(estimates, best_candidate(opt_result))
    ProgressMeter.next!(p)
end

@save joinpath(DATA_DIR, "params.jld2") estimates
