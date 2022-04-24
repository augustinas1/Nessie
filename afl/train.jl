using JLD2

include("../train_NN.jl")
include("afl.jl")

# Load training, validation and test datasets

@load joinpath(AFL_DIR, "train_data.jld2") X_train y_train
@load joinpath(AFL_DIR, "valid_data.jld2") X_valid y_valid
@load joinpath(AFL_DIR, "test_data.jld2") X_test y_test

train_data = (X_train, y_train)
valid_data = (X_valid, y_valid)
test_data = (X_test, y_test)

# Build the neural network

build_model(n_comps::Int, x::Int) = build_model(n_comps, [x])

function build_model(n_comps::Int, x::Vector{Int}=[32])
    hidden_layers = [Dense(x[i-1], x[i], relu) for i in 2:length(x)]
    model = Chain(InputLayer(),
                  Dense(1 + numparams(rn_afl), x[1], relu),
                  hidden_layers...,
                  MNBOutputLayer(x[end], n_comps)
            )
    MNBModel(model)
end

model = build_model(4, 128)

# Training

@time train_losses, valid_losses = train_NN!(model, train_data, valid_data; max_rounds=500, lr=0.01, batchsize=64)

println("Training dataset")
println("KLD: ", mean_loss(X_train, y_train, model; loss=loss_kldivergence))
println("Hellinger: ", mean_loss(X_train, y_train, model; loss=loss_hellinger))

println("\nValidation dataset")
println("KLD: ", mean_loss(X_valid, y_valid, model; loss=loss_kldivergence))
println("Hellinger: ", mean_loss(X_valid, y_valid, model; loss=loss_hellinger))

println("\nTest dataset")
println("KLD: ", mean_loss(X_test, y_test, model; loss=loss_kldivergence))
println("Hellinger: ", mean_loss(X_test, y_test, model; loss=loss_hellinger))

#plot([train_losses valid_losses], label=["training" "validation"])

@save joinpath(AFL_DIR, "model.jld2") model