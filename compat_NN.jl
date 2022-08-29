# This file contains various performance hacks mostly involving Zygote.
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

