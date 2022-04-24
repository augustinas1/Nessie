using JLD2

include("../train_NN.jl")
include("model.jl")
include("../viz.jl")

@load joinpath(MODEL_DIR, "train_data.jld2") X_train y_train
@load joinpath(MODEL_DIR, "valid_data.jld2") X_valid y_valid

# considering time snapshots t = {2, 4, 10, 16, 32, 50, 74, 100}
# training data saved for t = 2:2:100
# validation/test data saved for t = 1:1:100
m = 50; l = length(X_train) 
inds = sort(vcat(1:m:l, 2:m:l, 5:m:l, 8:m:l, 16:m:l, 25:m:l, 37:m:l, 50:m:l))
X_train = X_train[inds]; y_train = y_train[inds]

m = 100; l = length(X_valid)
inds = sort(vcat(2:m:l, 4:m:l, 10:m:l, 16:m:l, 32:m:l, 50:m:l, 74:m:l, m:m:l))
X_valid = X_valid[inds]; y_valid = y_valid[inds]

train_data = (X_train, y_train)
valid_data = (X_valid, y_valid)
GC.gc()

function build_model(n_comps::Int, x::Vector{Int}=[32])
    hidden_layers = [Dense(x[i-1], x[i], relu) for i in 2:length(x)]
    model = Chain(InputLayer(),
                  Dense(1 + numparams(rn), x[1], relu),
                  hidden_layers...,
                  MNBOutputLayer(x[end], n_comps)
            )
    MNBModel(model)
end

model = build_model(6, [1024])

#display(model.nn)

@time train_losses, valid_losses = train_NN!(model, train_data, valid_data; max_rounds=500, lr=0.001, batchsize=1000)
@save joinpath(MODEL_DIR, "model.jld2") model

println("KL divergence")
println(mean_loss(train_data..., model; loss=loss_kldivergence))
println(mean_loss(valid_data..., model; loss=loss_kldivergence))
println("Hellinger distance")
println(mean_loss(train_data..., model; loss=loss_hellinger))
println(mean_loss(valid_data..., model; loss=loss_hellinger))

# clean memory as it gets hogged up
GC.gc() 