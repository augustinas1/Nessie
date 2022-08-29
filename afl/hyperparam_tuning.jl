using JLD2

include("../train_NN.jl")
include("../viz.jl")
include("../generate_data.jl")
include("afl.jl")

# load validation data
@load joinpath(AFL_DIR, "valid_data.jld2") X_valid y_valid
valid_data = (X_valid, y_valid)

# load neural network
@load joinpath(AFL_DIR, "model.jld2") model

# FSP solver
fsp_sys = FSPSystem(rn_afl, combinatoric_ratelaw=false)
state_space = [2, 400]
u0 = zeros(state_space...)
u0[2, 1] = 1.0
fsp_prob = convert(ODEProblem, fsp_sys, u0, (0., 1.), ones(numparams(rn_afl)))
solver(ts, p) = fsp_solve(fsp_prob, ts, p; marginals=[2], abstol=1e-6, reltol=1e-6)

# ---------------------------------------------------------------------------------------------------
# Size of the dataset
# ---------------------------------------------------------------------------------------------------

nsamples = 10
nparams = [50, 100, 200, 300, 400, 500, 750, 1000]

models = []
@time for nparam in nparams
    
    _models = []

    for i in 1:nsamples

        tpts = [ Sobol.next!(s) for i in 1:nparam]
        _train = build_dataset_parallel(ts, tpts, solver)
        
        vpts = [ Sobol.next!(s) for i in 1:100]
        _valid = build_dataset_parallel(ts, vpts, solver)
        
        model = build_model(4, 128)
        _, _ = train_NN!(model, _train, _valid; max_rounds=500, lr=0.01, batchsize=64)
        push!(_models, model)
    end
    
    push!(models, _models)

end

#@save joinpath(AFL_DIR, "models_nparams.jld2") models
@load joinpath(AFL_DIR, "models_nparams.jld2") models

losses_hell = [ [ mean_loss(valid_data..., model; loss=loss_hellinger) for model in models[i] ] for i in 1:length(models)]

x = nparams
y = mean.(losses_hell)
ystd = std.(losses_hell)
ind = findmax(y)[2]
ymax = (y[ind] + ystd[ind]) * 1.01
xmax = x[end] * 1.01

plt_hell = plot(x, y, yerror=ystd, xlabel="dataset size", ylabel="Hellinger distance", 
                markerstrokecolor = colorant"#0088c3ff", leg=false, grid=false, lw=1.5, ylim=(0., ymax), xlim = (0., xmax),
                markerstrokewidth=1.5, tick_direction=:out, c=colorant"#0088c3ff",
                size = (290, 130), guidefontsize=6, tickfontsize=6, thickness_scaling=1.0,
                left_margin=-1Plots.mm, right_margin=0Plots.mm, top_margin=-1Plots.mm, bottom_margin=0Plots.mm)

savefig(joinpath(AFL_DIR, "hell_vs_nparams.svg"))


# ---------------------------------------------------------------------------------------------------
# Number of mixture components
# ---------------------------------------------------------------------------------------------------

nsamples = 10
ncomps = 1:10

models = []
@time for ncomp in ncomps
    
    _models = []
    
    for i in 1:nsamples
         tpts = [ Sobol.next!(s) for i in 1:1000]
        _train = build_dataset_parallel(ts, tpts, solver)
        
        vpts = [ Sobol.next!(s) for i in 1:100]
        _valid = build_dataset_parallel(ts, vpts, solver)
        
        model = build_model(ncomp, 128)
        _, _ = train_NN!(model, _train, _valid; max_rounds=500, lr=0.01, batchsize=64)
        push!(_models, model)
    end
    
    push!(models, _models)

end

#@save joinpath(AFL_DIR, "models_ncomps.jld2") models
@load joinpath(AFL_DIR, "models_ncomps.jld2") models

losses_hell = [ [ mean_loss(valid_data..., model; loss=loss_hellinger) for model in models[i] ] for i in 1:length(models)]

x = ncomps
y = mean.(losses_hell)
ystd = std.(losses_hell)
ind = findmax(y)[2]
ymax = (y[ind] + ystd[ind]) * 1.01

plt_hell = plot(x, y, yerror=ystd, xlabel="number of mixture components", ylabel="Hellinger distance", 
                markerstrokecolor = colorant"#0088c3ff", leg=false, grid=false, lw=1.5, ylim=(0., ymax),
                markerstrokewidth=1.5, tick_direction=:out, c=colorant"#0088c3ff", xticks=x,
                size = (290, 130), guidefontsize=6, tickfontsize=6, thickness_scaling=1.0,
                left_margin=-1Plots.mm, right_margin=-2Plots.mm, top_margin=-1Plots.mm, bottom_margin=0Plots.mm)

savefig(joinpath(AFL_DIR, "hell_vs_ncomps.svg"))


# ---------------------------------------------------------------------------------------------------
# Number of neurons
# ---------------------------------------------------------------------------------------------------

nsamples = 10
nunits = 2 .^(1:10)

models = []
@time for nunit in nunits
    
    _models = []
    
    for i in 1:nsamples
         tpts = [ Sobol.next!(s) for i in 1:1000]
        _train = build_dataset_parallel(ts, tpts, solver)
        
        vpts = [ Sobol.next!(s) for i in 1:100]
        _valid = build_dataset_parallel(ts, vpts, solver)
        
        model = build_model(4, nunit)
        _, _ = train_NN!(model, _train, _valid; max_rounds=500, lr=0.01, batchsize=64)
        push!(_models, model)
    end
    
    push!(models, _models)

end

#@save joinpath(AFL_DIR, "models_nunits.jld2") models
@load joinpath(AFL_DIR, "models_nunits.jld2") models

losses_hell = [ [ mean_loss(valid_data..., model; loss=loss_hellinger) for model in models[i] ] for i in 1:length(models)]

x = string.(nunits)
y = mean.(losses_hell)
ystd = std.(losses_hell)
ind = findmax(y)[2]
ymax = (y[ind] + ystd[ind]) * 1.01

plt_hell = plot(x, y, yerror=ystd, xlabel="number of neurons", ylabel="Hellinger distance",
                markerstrokecolor = colorant"#0088c3ff", leg=false, grid=false, lw=1.5, ylim=(0., ymax),
                markerstrokewidth=1.5, tick_direction=:out, c=colorant"#0088c3ff",
                size = (290, 130), guidefontsize=6, tickfontsize=6, thickness_scaling=1.0,
                left_margin=-1Plots.mm, right_margin=-1Plots.mm, top_margin=-1Plots.mm, bottom_margin=0Plots.mm)

savefig(joinpath(AFL_DIR, "hell_vs_nunits.svg"))


# ---------------------------------------------------------------------------------------------------
# Different deep network architectures
# ---------------------------------------------------------------------------------------------------

nlayers = [[128], [128, 128], [64, 16], [128, 64, 32], [64, 32, 16], [64, 64, 32, 32], [16, 16, 16, 16]]
nsamples = 10

models = []
@time for nlayer in nlayers
    
    _models = []
    
    for i in 1:nsamples
         tpts = [ Sobol.next!(s) for i in 1:1000]
        _train = build_dataset_parallel(ts, tpts, solver)
        
        vpts = [ Sobol.next!(s) for i in 1:100]
        _valid = build_dataset_parallel(ts, vpts, solver)
        
        model = build_model(4, nlayer)
        _, _ = train_NN!(model, _train, _valid; max_rounds=500, lr=0.01, batchsize=64)
        push!(_models, model)
    end
    
    push!(models, _models)

end

#@save joinpath(AFL_DIR, "models_nlayers.jld2") models
@load joinpath(AFL_DIR, "models_nlayers.jld2") models

losses_hell = [ [ mean_loss(valid_data..., model; loss=loss_hellinger) for model in models[i] ] for i in 1:length(models)]
labels = ["128", "128-128", "64-16", "128-64-32", "64-32-16", "64-64-32-32", "16-16-16-16"]

plt_hell = plot(labels, mean.(losses_hell), yerror=std.(losses_hell), 
                xlabel="hidden layer architecture", ylabel="Hellinger distance", 
                markerstrokecolor = colorant"#0088c3ff", leg=false, grid=false, lw=1.5,
                markerstrokewidth=1.5, tick_direction=:out, c=colorant"#0088c3ff",
                size = (290, 130), guidefontsize=6, tickfontsize=6, thickness_scaling=1.0, xrotation = 30,
                left_margin=-1Plots.mm, right_margin=-2Plots.mm, top_margin=-1.5Plots.mm, bottom_margin=3Plots.mm)

                savefig(joinpath(AFL_DIR, "hell_vs_architecture.svg"))


# ---------------------------------------------------------------------------------------------------
# Number of SSA trajectories
# ---------------------------------------------------------------------------------------------------

# set up SSA
u0 = [1, 0] # [G, P]
jsys = convert(JumpSystem, rn_afl, combinatoric_ratelaws=false)
dprob = DiscreteProblem(jsys, u0, (0., 1.), ones(numparams(rn_afl)))
jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false,false))

ntrajs = Int.([1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4])

models = []
nsamples = 10

@time for ntraj in ntrajs
    
    ssa_solver(ts, p) = ssa_solve(jprob, ts, p, ntraj; marginals=[2])
    _models = []
    
    for i in 1:nsamples
        tpts = [ Sobol.next!(s) for i in 1:1000 ]
        _train = build_dataset(ts, tpts, ssa_solver) # SSA
        
        vpts = [ Sobol.next!(s) for i in 1:100 ]
        _valid = build_dataset_parallel(ts, vpts, solver) # FSP
        
        model = build_model(4, 128)
        _, _ = train_NN!(model, _train, _valid; max_rounds=500, lr=0.01, batchsize=64)
        push!(_models, model)
    end
    
    push!(models, _models)

end

#@save joinpath(AFL_DIR, "models_ntrajs.jld2") models

# add FSP as a purple dashed line
# using the 1st model from "models_nlayers.jld2" as that's we need
@load joinpath(AFL_DIR, "models_nlayers.jld2") models
fsp_res = mean([ mean_loss(valid_data..., model; loss=loss_hellinger) for model in models[1] ])
_x = [ntrajs[1]-0.2, ntrajs..., ntrajs[end]*1.3]; _y = fill(fsp_res,length(_x))
plt_hell = plot(_x, _y, linestyle=:dash, lw=1.0, linealpha=1.0, linecolor=colorant"#a93a45")

@load joinpath(AFL_DIR, "models_ntrajs.jld2") models
losses_hell = [ [ mean_loss(valid_data..., model; loss=loss_hellinger) for model in models[i] ] for i in 1:length(models)]
y = mean.(losses_hell)
ystd = std.(losses_hell)
ind = findmax(y)[2]
ymax = (y[ind] + ystd[ind]) * 1.05

plt_hell = plot!(plt_hell, ntrajs, y, yerror=ystd, ylim = (0., ymax), c=colorant"#0088c3ff",
                xlabel="number of SSA samples", ylabel="Hellinger distance", legend=false, xlim=(ntrajs[1]-0.2, ntrajs[end]*1.2),
                markerstrokecolor = colorant"#0088c3ff", markerstrokewidth=1.5, grid=false, lw=1.5, xaxis=:log,
                size=(190, 130),  guidefontsize=6, tickfontsize=6, thickness_scaling=1.0,
                left_margin=-2Plots.mm, bottom_margin=-1.5Plots.mm, right_margin=-1Plots.mm, top_margin=-2Plots.mm, tick_direction=:out)

savefig(joinpath(AFL_DIR, "hell_vs_ssa.svg"))

# ---------------------------------------------------------------------------------------------------
# Noisiness in SSA histograms
# ---------------------------------------------------------------------------------------------------

@load joinpath(AFL_DIR, "models_ntrajs.jld2") models
p = Float32.([1.17, 0.05, 6.9, 47.9]); t = 10f0
ps = vcat(t, p)
ylim = 0.052
xlim = 65
xticks = 0:20:60

_solver(t, p) = ssa_solve(jprob, t, p, 100; marginals=[2])
_data1 = build_dataset(t, [p], _solver)

dif = xlim+1-length(_data1[2][1])
if dif > 0
    _data1[2][1] = vcat(_data1[2][1], zeros(Float32, dif))
end
plt1 = plot_dist(ps, _data1, models[5][rand(1:10)]; true_label="SSA", title="", xticks=xticks)
plt1 = plot!(plt1, xlabel="", yticks=false, ylabel="Probability", left_margin=0Plots.mm)
plt1 = plot!(plt1, ylims=(0., ylim), xlims=(0., xlim), leg=false)
plt1 = annotate!(plt1, [(30, ylim-0.002, Plots.text("10²", 8, :black, :center))])

_solver(t, p) = ssa_solve(jprob, t, p, 1000; marginals=[2])
_data2 = build_dataset(t, [p], _solver)
dif = xlim+1-length(_data2[2][1])
if dif > 0
    _data2[2][1] = vcat(_data2[2][1], zeros(Float32, dif))
end
plt2 = plot_dist(ps, _data2, models[7][rand(1:10)], xticks=xticks, true_label="SSA")
plt2 = plot!(plt2, xlabel="Protein number", ylabel="", title="", leg=true, yticks=false, foreground_color_legend = nothing) 
plt2 = plot!(plt2, ylims=(0., ylim), xlims=(0., xlim))
plt2 = annotate!(plt2, [(33, ylim-0.002, Plots.text("10³", 8, :black, :center))])

_solver(t, p) = ssa_solve(jprob, t, p, 10000; marginals=[2])
_data3 = build_dataset(t, [p], _solver)
dif = xlim+1-length(_data3[2][1])
if dif > 0
    _data3[2][1] = vcat(_data3[2][1], zeros(Float32, dif))
end
plt3 = plot_dist(ps, _data3, models[9][rand(1:10)], xticks=xticks)
plt3 = plot!(plt3, xlabel="", ylabel="", title="", leg=false, yticks=false) 
plt3 = plot!(plt3, ylims=(0., ylim), xlims=(0., xlim))
plt3 = annotate!(plt3, [(33, ylim-0.002, Plots.text("10⁴", 8, :black, :center))])

plt = plot(plt1, plt2, plt3, layout=(1,3), size=(400, 130), legendfontsize=6,
           guidefontsize=6, tickfontsize=6, ticksize=6, thickness_scaling=1.0, tick_orientation=:out,
           top_margin=-1Plots.mm, bottom_margin=0.5Plots.mm, left_margin=1.5Plots.mm, right_margin=-2Plots.mm)

savefig(joinpath(AFL_DIR, "SSA_dists.svg"))
