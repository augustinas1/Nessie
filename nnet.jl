using Flux, Flux.Data
using SpecialFunctions: logbeta
using StatsBase, Distributions
using Base: @kwdef

##

# Standard split functionality from the Flux documentation (under "Advanced Model Building")
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)
Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

##

""" 
    Convenience struct for the input layers used in the paper, which do nothing but log-transform
    their inputs.
"""
struct InputLayer end

(::InputLayer)(x::AbstractArray) = log.(x)

"""
    Construct an output layer for a MNB neural network. The output is a tuple of three vectors:
    the count parameters `r`, the success probabilities `p` and the mixture weights `w`.
"""
function MNBOutputLayer(x, n_comps)
    layer_ww = Chain(Dense(x, n_comps), softmax)
    layer_pp = Dense(x, n_comps, sigmoid)
    layer_rr = Dense(x, n_comps, exp)

    Split(layer_rr, layer_pp, layer_ww)
end

##

"""
    Lightweight convenience wrapper for a neural network that outputs mixtures of negative binomials.
    Mainly used in connection with the utility functions defined in this section.
"""
struct MNBModel{T}
    nn::T
end

Flux.@functor MNBModel

(m::MNBModel)(x::AbstractArray) = m.nn(x)

Distribution(m::MNBModel, x::AbstractVector) = NegativeBinomialMixture(m, x)

function NegativeBinomialMixture(m::MNBModel, x::AbstractVector)
    rr, pp, ww = m(x)
    NegativeBinomialMixture(NegativeBinomial.(rr, pp), Categorical(ww))
end

StatsBase.mean(m::MNBModel, x::AbstractVector) = mean(Distribution(m, x))
StatsBase.var(m::MNBModel, x::AbstractVector) = var(Distribution(m, x))
StatsBase.std(m::MNBModel, x::AbstractVector) = std(Distribution(m, x))
Distributions.pdf(m::MNBModel, x::AbstractVector, k) = pdf(Distribution(m, x), k)
Distributions.logpdf(m::MNBModel, x::AbstractVector, k) = logpdf(Distribution(m, x), k)

##

""" 
    Optimised version of `Distributions.logpdf(NegativeBinomial(r, p), k)`. 
"""
function nblogpdf(r, p, k)
    # mostly copy from Distributions.jl NegativeBinomial def
    # iszero(p) && @warn "p = 0 (k = $k, r = $r)"
    # isone(p) && @warn "p = 1 (k = $k, r = $r)"
    # iszero(r) && @warn "r = 0 (k = $k, p = $p)"
    p_mod = oftype(p, p * 0.999999f0)
    r * log(p_mod) + k * log1p(-p_mod) - log(k + r) - logbeta(r, k + 1)
end

nbpdf(r, p, k) = exp(nblogpdf(r, p, k))

"""
    Optimised version of `Distributions.pdf(NegativeBinomialMixture(...), k)`.
"""
function mix_nbpdf(rr::AbstractVector, pp::AbstractVector,
                   ww::AbstractVector, k)
    ret = ww[1] .* nbpdf.(rr[1], pp[1], k)
    
    @inbounds for i in 2:length(ww)
        ret = ret .+ ww[i] .* nbpdf.(rr[i], pp[i], k)
    end
    
    ret
end
