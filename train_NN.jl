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

# Don't use this for training the NN yet!

loss_wass2(x, y, model) = loss_wasserstein(x, y, model; p=2)

function loss_wasserstein(x::AbstractVector, y::AbstractVector, model; p=1)
    dist = Distribution(model, x)
    
    nmax = ceil(Int, max(length(y) - 1, mean(dist) + 3 * std(dist)))
    xx = 0:nmax
    
    pred = pdf.(dist, xx)
    pred ./= sum(pred)
    
    wasserstein(DiscreteNonParametric(xx, pred), 
                DiscreteNonParametric(0:length(y) - 1, y), p=p)
end

## Loss utility functions
function batch_loss(X::AbstractVector, y::AbstractVector, model;
                    loss = loss_crossentropy)
    ret = loss(X[1], y[1], model)
    
    @inbounds for i in 2:length(X)
        ret += loss(X[i], y[i], model)
    end
    
    ret / length(X)
end

function mean_loss(X::AbstractVector, y::AbstractVector, model;
                    loss = loss_crossentropy)
    ret = zeros(Float32, Threads.nthreads())
    
    Threads.@threads for i in 1:length(X)
        ret[Threads.threadid()] += loss(X[i], y[i], model)
    end
    
    sum(ret) / length(X)
end

sqnorm(x) = sum(abs2, x)
l2_loss(p) = sum(sqnorm, p)

##

struct TrainArgs{OT,DT}
    # Hyperparameters
    lr::Float64
    l2_reg::Float64
    max_rounds::Int
    min_lr::Float64
    batchsize::Int
    optimizer::Type{OT}
    
    train_data::DT
    valid_data::DT
end

function TrainArgs(train_data, valid_data, optimizer;
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

mutable struct NNTrainer{DL,MT,OT,AT <: TrainArgs}
    train_loader::DL
    
    train_losses::Vector{Float32}
    valid_losses::Vector{Float32}
    
    lr_updates::Vector{Int}
    args::AT
    model::MT
    opt::OT
end
    
function NNTrainer(args::TrainArgs, model)
    train_loader = DataLoader(args.train_data, batchsize=args.batchsize, shuffle=true)
    
    trainer = NNTrainer(train_loader, Float32[], Float32[], [1], args, model, args.optimizer(args.lr))
    update_losses!(trainer)
    trainer
end

function update_losses!(trainer::NNTrainer) 
    train_loss = mean_loss(trainer.args.train_data[1], trainer.args.train_data[2], trainer.model; loss=loss_kldivergence)
    valid_loss = mean_loss(trainer.args.valid_data[1], trainer.args.valid_data[2], trainer.model; loss=loss_kldivergence)
    
    push!(trainer.train_losses, train_loss)
    push!(trainer.valid_losses, valid_loss)
    nothing
end

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

function should_decrease_lr(trainer::NNTrainer)
    losses = trainer.valid_losses
    
    d = 50 # history length param
    round = length(losses)
    round <= last(trainer.lr_updates) + d && return false
    
    mean(losses[end-20:end]) > mean(losses[end-d:end-20]) * 0.995
end

function train_NN!(model, train_data, valid_data; 
                   optimizer = ADAM, cb=nothing, threads=true, loss=loss_crossentropy, kwargs...)
    args = TrainArgs(train_data, valid_data, optimizer; kwargs...)

    ## Progress meter
    progress = if !(cb isa Nothing)
        nothing
    else
        Progress(args.max_rounds; dt=1, desc="Training...")
    end
    
    trainer = NNTrainer(args, model)
    
    for iter in trainer
        train_round!(trainer, threads; loss)

        cb !== nothing && cb(trainer.opt.eta, trainer.train_losses[end], trainer.valid_losses[end])
        
        ProgressMeter.next!(progress; showvalues = [(:iter, iter), 
                                                    (:learning_rate, trainer.opt.eta), 
                                                    (:train_loss, trainer.train_losses[end]),
                                                    (:valid_loss, trainer.valid_losses[end])])
    end

    progress !== nothing && finish!(progress)
    
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