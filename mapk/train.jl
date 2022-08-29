# Train Nessie using the training and validation sets
# generated using gen.jl, saving the resulting network
# parameters in model.jld2
using JLD2
using Random
using ProgressMeter
using StatsBase
using Flux

include("../train_NN.jl")

include("network.jl")
include("inference.jl")

@load joinpath(DATA_DIR, "train_data.jld2") X_train y_train
@load joinpath(DATA_DIR, "val_data.jld2") X_val y_val

train_data = (X_train, y_train)
pretrain_data = (X_train[1:12000], y_train[1:12000])
val_data = (X_val, y_val);

model = build_model(5)

# Pretraining
@time train_losses, val_losses = train_NN!(model, pretrain_data, val_data; max_rounds=100, lr=0.001, batchsize=1000)
model_params = Flux.params(model)
@save joinpath(DATA_DIR, "model.jld2") model_params train_losses val_losses

# Training
@time train_losses, val_losses = train_NN!(model, train_data, val_data; max_rounds=400, lr=0.0001, batchsize=1000)
model_params = Flux.params(model)
@save joinpath(DATA_DIR, "model.jld2") model_params train_losses val_losses
