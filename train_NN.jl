# TODO: minimise allocations in calculating the loss and evaluate batches as matrices somehow?

using Flux, Flux.Data
using SpecialFunctions: logbeta
using Distances
using StatsBase, Distributions
using OptimalTransport
using Base: @kwdef

using ProgressMeter
ProgressMeter.ijulia_behavior(:append)
ProgressMeter.ijulia_behavior(:clear)

include("nbmixture.jl")
include("nnet.jl")
include("compat_NN.jl")

"""
    Compute the predicted distribution at input location `x` and 
    evaluate its pdf at points `yy`. Faster alternative to
    `pdf(Distribution(model, x), yy)` for `MNBModel`s.
"""
function pred_pdf(model, x::AbstractVector, yy)
    rr, pp, ww = model(x)
    mix_nbpdf(rr, pp, ww, yy)
end

## Loss functions

function loss_kldivergence(x::AbstractVector, y::AbstractVector, model)
    pred = pred_pdf(model, x, 0:length(y)-1)
    Flux.kldivergence(pred, y)
end

function loss_reversekldivergence(x::AbstractVector, y::AbstractVector, model)
    pred = pred_pdf(model, x, 0:length(y)-1)
    Flux.kldivergence(y, pred)
end

# Equals KL divergence + const.
function loss_crossentropy(x::AbstractVector, y::AbstractVector, model)
    pred = pred_pdf(model, x, 0:length(y)-1)
    Flux.crossentropy(pred, y)
end

function loss_hellinger(x::AbstractVector, y::AbstractVector, model)
    pred = pred_pdf(model, x, 0:length(y)-1)
    hellinger(Float64.(pred), Float64.(y))
end

## Loss utility functions

""" 
    Computes the average loss over a batch. Here `X` is a vector of inputs
    and `y` is the corresponding vector of outputs.
"""
function batch_loss(X::AbstractVector, y::AbstractVector, model;
                    loss = loss_crossentropy)
    ret = loss(X[1], y[1], model)
    
    @inbounds for i in 2:length(X)
        ret += loss(X[i], y[i], model)
    end
    
    ret / length(X)
end

"""
    Similar to `batch_loss`, but multi-threaded.
"""
function mean_loss(X::AbstractVector, y::AbstractVector, model;
                    loss = loss_crossentropy)
    ret = zeros(Float32, Threads.nthreads())
    
    Threads.@threads for i in 1:length(X)
        ret[Threads.threadid()] += loss(X[i], y[i], model)
    end
    
    sum(ret) / length(X)
end

# For regularisation
sqnorm(x) = sum(abs2, x)
l2_loss(p) = sum(sqnorm, p)

##

""" 
    Wrapper struct for training hyperparameters
"""
struct TrainArgs{OT,DT}
    lr::Float64             # Current learning rate
    l2_reg::Float64         # L2 regularisation weight
    max_rounds::Int         # Maximum number of epochs
    min_lr::Float64         # Minimum learning rate 
    batchsize::Int          # Batch size
    optimizer::Type{OT}     # Optimizer (e.g. `Flux.ADAM`)
    
    train_data::DT          # Training dataset
    valid_data::DT          # Validation dataset
end

function TrainArgs(train_data, valid_data, optimizer = ADAM;
                   lr::Real, 
                   max_rounds::Int, 
                   batchsize::Int = 100,
                   l2_reg = 0, 
                   min_lr::Real = lr / 32) 
    batchsize == 0 && (batchsize = length(train_data[1]))
    TrainArgs(Float64(lr), Float64(l2_reg), max_rounds, Float64(min_lr), batchsize, optimizer, 
              train_data, valid_data)
end

##

"""
    This struct captures some of the data generated while training Nessie.
"""
mutable struct NNTrainer{DL,MT,OT,AT <: TrainArgs}
    train_loader::DL                # Uses the `Flux.DataLoader` interface
    
    train_losses::Vector{Float32}   # Training loss at each epoch
    valid_losses::Vector{Float32}   # Validation loss at each epoch
        
    lr_updates::Vector{Int}         # Epochs at which the learning rate was updated
    args::AT                        # Training arguments
    model::MT                       # Model to train
    opt::OT                         # Optimiser
end
    
function NNTrainer(args::TrainArgs, model)
    train_loader = DataLoader(args.train_data, batchsize=args.batchsize, shuffle=true)
    
    trainer = NNTrainer(train_loader, Float32[], Float32[], [1], args, model, args.optimizer(args.lr))
    update_losses!(trainer)
    trainer
end

"""
    Compute training & validation losses at the current epoch and save them in the training struct.
"""
function update_losses!(trainer::NNTrainer) 
    train_loss = mean_loss(trainer.args.train_data[1], trainer.args.train_data[2], trainer.model; loss=loss_kldivergence)
    valid_loss = mean_loss(trainer.args.valid_data[1], trainer.args.valid_data[2], trainer.model; loss=loss_kldivergence)
    
    push!(trainer.train_losses, train_loss)
    push!(trainer.valid_losses, valid_loss)
    nothing
end

"""
    Iteration facilities for the trainer. Each iteration increases the current round by 1 and
    checks if the learning rate should be decreased at that round; if so, it changes the learning
    rate. This function can mutate the trainer!
"""
Base.iterate(trainer::NNTrainer) = (length(trainer.train_losses), trainer)

function Base.iterate(iter, trainer::NNTrainer)
    iter = length(trainer.train_losses)
    iter < trainer.args.max_rounds || return nothing
    
    if should_decrease_lr(trainer)
        new_lr = trainer.opt.eta / 2
        new_lr >= trainer.args.min_lr || return nothing
        
        push!(trainer.lr_updates, iter)
        trainer.opt = trainer.args.optimizer(new_lr)
    end
    
    (iter + 1, trainer)
end

##

"""
    Decrease the learning rate if at least 50 rounds have passed since the 
    last decrease, and if the mean validation loss has changed by less than 0.5%
    in the last 50 rounds.
"""
function should_decrease_lr(trainer::NNTrainer)
    losses = trainer.valid_losses
    
    round = length(losses)
    round <= last(trainer.lr_updates) + 50 && return false
    
    mean(losses[end-25:end]) > mean(losses[end-50:end-25]) * 0.995
end

##

"""
    train_NN!(model, train_data, valid_data; kwargs...)

    Train Nessie using the given training data and validation data. `train_data` and `valid_data` should be tuples
    `(X, y)`, where `X` is a vector of input points and `y` the corresponding vector of training data. Returns
    the training and validation losses for each epoch.

    The following keyword arguments are supported by this function:
        `threads`: use multithreading (defaults to `true`)
        `loss`: loss function to use (defaults to `loss_crossentropy`)

    All other keyword arguments will be passed to `TrainArgs` (see above).
"""
function train_NN!(model, train_data, valid_data; 
                   threads=true, loss=loss_crossentropy, kwargs...)
    args = TrainArgs(train_data, valid_data; kwargs...)

    ## Progress meter
    progress = Progress(args.max_rounds; dt=1, desc="Training...")
    
    trainer = NNTrainer(args, model)
    
    for iter in trainer
        train_round!(trainer, threads; loss)

        ProgressMeter.next!(progress; showvalues = [(:iter, iter), 
                                                    (:learning_rate, trainer.opt.eta), 
                                                    (:train_loss, trainer.train_losses[end]),
                                                    (:valid_loss, trainer.valid_losses[end])])
    end

    finish!(progress)
    
    trainer.train_losses, trainer.valid_losses
end

macro maybe_threaded(ex)
    if Threads.nthreads() == 1
        return esc(ex)
    else
        return esc(:(if threads 
                        Threads.@threads $ex
                    else
                        $ex
                    end))
    end
end

"""
    Perform one training epoch.
"""
function train_round!(trainer::NNTrainer, threads::Bool=true; loss=loss_cross_entropy)
    p = Flux.params(trainer.model)
    
    nt = Threads.nthreads()
    grads = Vector{Flux.Zygote.Grads}(undef, nt)
         
    for (x, y) in trainer.train_loader
        @maybe_threaded for i in 1:nt
            grads[i] = Flux.gradient(p) do
                batch_loss((@view x[i:nt:end]), (@view y[i:nt:end]), trainer.model; loss)
            end
        end

        grad_total = reduce(.+, grads)
        if trainer.args.l2_reg != 0
            grad_total .+= Flux.gradient(() -> trainer.args.l2_reg * l2_loss(p), p)
        end
        
        Flux.Optimise.update!(trainer.opt, p, grad_total)
    end
    
    update_losses!(trainer)
end
