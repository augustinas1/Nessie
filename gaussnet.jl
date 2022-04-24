using Flux, Distributions

struct MGModel{T}
    nn::T
end

Flux.@functor MGModel

(m::MGModel)(x::AbstractArray) = m.nn(x)

Distribution(m::MGModel, x::AbstractVector) = GaussianMixture(m, x)

function GaussianMixture(m::MGModel, x::AbstractVector)
    mm, ss, ww = m(x)
    MixtureModel(Normal.(mm, ss), Categorical(ww))
end

StatsBase.mean(m::MGModel, x::AbstractVector) = mean(Distribution(m, x))
StatsBase.var(m::MGModel, x::AbstractVector) = var(Distribution(m, x))
StatsBase.std(m::MGModel, x::AbstractVector) = std(Distribution(m, x))
Distributions.pdf(m::MGModel, x::AbstractVector, k) = pdf(Distribution(m, x), k)
Distributions.logpdf(m::MGModel, x::AbstractVector, k) = logpdf(Distribution(m, x), k)

##

function MGOutputLayer(x, n_comps; s_min = 0)
    layer_ww = Chain(Dense(x, n_comps), softmax)
    layer_mm = Dense(x, n_comps, exp)
    layer_ss = Dense(x, n_comps, inp -> exp(inp) + s_min)

    Split(layer_mm, layer_ss, layer_ww)
end

function pred_pdf(model::MGModel, x::AbstractVector, yy)
    pdf(Distribution(model, x), yy)
end

#@plain_struct MGOutputLayer
@plain_struct MGModel