using Flux

d = 11
intensity = 1
prior = Product(Uniform.(logranges[1:d, 1], logranges[1:d, 2]))

###
### Note: the use of anonymous functions here may break saving and loading
### the neural network using JLD2, depending on the Julia version.
### The safest option is to save network weights instead.
###
function ZIMNBOutputLayer(x, n_comps)
    layer_ww = Chain(Dense(x, n_comps), softmax)
    layer_pp = Chain(Dense(x, n_comps-1, sigmoid), x -> [ x; 1e-3 ])
    layer_rr = Chain(Dense(x, n_comps-1, exp), x -> [ x; 0.999 ])

    Split(layer_rr, layer_pp, layer_ww)
end

function build_model(n_comps::Int)
    # Inputs to the network are (t - time variable, θ... - rate parameters)
    # outpus are the r, p, and weight parameters characterising the mixture of k negative binomials
    x = 2048
    model = Chain(InputLayer(),
                  Dense(12, x, relu),
                  ZIMNBOutputLayer(x, n_comps)
            )
    MNBModel(model)
end

yobs = [ cell_obs[j+1] for j in 1:length(tt) ]
hist_yobs = [ Float64[ count(yobs[i] .<= 100); [ count(j .<= yobs[i] .< j + 1) for j in 100:2000 ]] for i in 1:8 ]
hist_yobs ./= sum.(hist_yobs)

function hellinger(aa, bb)
    ret = 0.0
    for (a, b) in zip(aa, bb)
        ret += (sqrt(a) - sqrt(b))^2
    end
    
    ret
end

function loss_hellinger_mapk(x::AbstractVector, model, tt=tt, yobs_clean=yobs_clean, yobs_bin0=yobs_bin0, bin_thresh=bin_thresh)
    bufs = zeros(Threads.nthreads())

    Threads.@threads for i in 2:length(tt)
        ps = [ tt[i]; 10 .^ x ]
        
        pred = pred_pdf(model, ps, 100:2000)
        pred[1] += cdf(model, ps, 99)
        
        bufs[Threads.threadid()] += hellinger(pred, hist_yobs[i])
    end
    
    sum(bufs)
end
