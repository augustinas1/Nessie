using ZygoteRules

function pullback_for_default_literal_getproperty(cx::ZygoteRules.AContext, x, ::Val{f}) where {f}
  return ZygoteRules._pullback(cx, ZygoteRules.literal_getfield, x, Val{f}())
end

macro plain_struct(typename)
    :( 
        function ZygoteRules._pullback(
          cx::ZygoteRules.AContext, ::typeof(ZygoteRules.literal_getproperty), x::$(typename), ::Val{f}
        ) where {f}
            return pullback_for_default_literal_getproperty(cx, x, Val{f}())
    end)
end

@plain_struct Dense
@plain_struct Chain
@plain_struct Split
@plain_struct MNBModel

# This stops Zygote from complaining about arrays being modified
function Distributions.var(d::NegativeBinomialMixture)
    K = ncomponents(d)
    p = probs(d)
    means = [ mean(component(d, i)) for i in 1:K ]
    m = 0.0
    v = 0.0
    for i = 1:K
        pi = p[i]
        ci = component(d, i)
        m += pi * means[i]
        v += pi * var(ci)
    end
    for i = 1:K
        pi = p[i]
        v += pi * abs2(means[i] - m)
    end
    return v
end


##

# Old NN training code
# function es_criterion(losses; min_epoch_size=30)
#     length(losses) <= min_epoch_size && return false
    
#     mean(losses[end-10:end]) > mean(losses[end-min_epoch_size:end-10]) * 0.995
# end

# function train_NN!(model, train_data, valid_data, train_losses=Float32[], valid_losses=Float32[];
#                    optimizer = ADAM, λ, min_epoch_size=30, max_rounds=5, cb=nothing, kws...)
#     args = NN_args(; kws...) # collect options in a struct for convenience


#     p = Flux.params(model) # model's trainable parameters

#     ## Progress meter
#     progress = if !(cb isa Nothing)
#         nothing
#     else
#         Progress(args.epochs; dt=1, desc="Training...")
#     end
    
#     push!(train_losses, mean_loss(train_data[1], train_data[2], model; loss=loss_kldivergence))
#     push!(valid_losses, mean_loss(valid_data[1], valid_data[2], model; loss=loss_kldivergence))
    
#     opt = optimizer(args.η)
    
#     es = Flux.plateau(() -> -length(train_losses), 2; min_dist=min_epoch_size + 1)
    
#     while length(train_losses) <= args.epochs
#         train_NN_round!(model, train_data, valid_data, progress; 
#                         train_losses, valid_losses, opt, λ, args, min_epoch_size, cb)
        
#         (opt.eta <= args.η / (2 ^ (max_rounds - 1)) || es()) && break
#         opt.eta /= 2
#     end

#     train_losses, valid_losses
# end

# ProgressMeter.next!(::Nothing; kwargs...) = nothing

# function train_NN_round!(model, train_data, valid_data, progress; 
#                          train_losses, valid_losses, opt, λ, args, min_epoch_size, cb=nothing)
#     p = Flux.params(model) # model's trainable parameters
    
#     train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    
#     nt = Threads.nthreads()
#     grads = Array{Any}(undef, nt)
    
#     epochs_done = length(train_losses)
    
#     ## Training
#     for epoch in epochs_done:args.epochs        
#         for (x, y) in train_loader
#             @Threads.threads for i in 1:nt
#                 grads[i] = Flux.gradient(p) do
#                     batch_loss((@view x[i:nt:end]), (@view y[i:nt:end]), model; loss=loss_crossentropy)
#                 end
#             end
                
#             grad_total = reduce(.+, grads) .+ Flux.gradient(() -> λ * l2_loss(p), p)
#             Flux.Optimise.update!(opt, p, grad_total) # update parameters
#         end
        
#         push!(train_losses, mean_loss(train_data[1], train_data[2], model; loss=loss_kldivergence))
#         push!(valid_losses, mean_loss(valid_data[1], valid_data[2], model; loss=loss_kldivergence))

#         ProgressMeter.next!(progress; showvalues = [(:iter, epoch), 
#                                                     (:learning_rate, opt.eta), 
#                                                     (:train_loss, train_losses[end]),
#                                                     (:valid_loss, valid_losses[end])])
        
#         cb !== nothing && cb(opt.eta, train_losses[end], valid_losses[end])
        
#         es_criterion(@view valid_losses[epochs_done:end]; min_epoch_size) && break
#     end
# end
