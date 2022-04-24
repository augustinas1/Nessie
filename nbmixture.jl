using Distributions, StatsBase
using SpecialFunctions
using Printf

##

struct NegativeBinomialMixture{T} <: AbstractMixtureModel{Univariate,Discrete,NegativeBinomial{T}}
    components::Vector{NegativeBinomial{T}}
    prior::Categorical{T,Vector{T}}
end

Distributions.ncomponents(mixture::NegativeBinomialMixture) = length(mixture.components)
Distributions.components(mixture::NegativeBinomialMixture) = mixture.components
Distributions.component(mixture::NegativeBinomialMixture, k::Int) = mixture.components[k]

Distributions.probs(mixture::NegativeBinomialMixture) = mixture.prior.p

StatsBase.rand(mixture::NegativeBinomialMixture) = rand(mixture.components[rand(mixture.prior)])

function Base.show(io::IO, d::NegativeBinomialMixture)
    K = ncomponents(d)
    pr = probs(d)
    println(io, "NegativeBinomialMixture(K = $K)")
    Ks = min(K, 8)
    for i = 1:Ks
        @printf(io, "components[%d] (prior = %.4f): ", i, pr[i])
        println(io, component(d, i))
    end
    if Ks < K
        println(io, "The rest are omitted ...")
    end
end

function StatsBase.kurtosis(dist::NegativeBinomialMixture)
    rr = [ comp.r for comp in dist.components ]
    pp = [ comp.p for comp in dist.components ]
    qq = 1 .- pp
    
    m3s = @. qq * (pp^2*rr + 3*pp*qq*rr*(rr+1) + qq^2*rr*(rr+1)*(rr+2)) / pp^3
    m4s = @. qq * (pp^3*rr + 7*pp^2*qq*rr*(rr+1) + 6*pp*qq^2*rr*(rr+1)*(rr+2) + qq^3*rr*(rr+1)*(rr+2)*(rr+3)) / pp^4
    
    m3 = sum(dist.prior.p .* m3s)
    m4 = sum(dist.prior.p .* m4s)
    
    m = mean(dist)
    s = std(dist)
    
    (m4 - 4*m*m3 + 6*m^2*(s^2+m^2) - 4*m^3*m + m^4) / s^4 - 3
end

function StatsBase.skewness(dist::NegativeBinomialMixture)
    rr = [ comp.r for comp in dist.components ]
    pp = [ comp.p for comp in dist.components ]
    qq = 1 .- pp
    
    m3s = @. qq * (pp^2*rr + 3*pp*qq*rr*(rr+1) + qq^2*rr*(rr+1)*(rr+2)) / pp^3
    
    m3 = sum(dist.prior.p .* m3s)
    
    m = mean(dist)
    s = std(dist)
    
    (m3 - 3*m*s^2 - m^3) / s^3
end

bimodcoeff(dist) = 1 / (kurtosis(dist) + 3 - skewness(dist) ^ 2)