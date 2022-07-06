using JLD2

include("../train_NN.jl")
include("../generate_data.jl")
include("model.jl")
include("../viz.jl")

@load joinpath(MODEL_DIR, "test_data.jld2") X_test y_test
test_data = (X_test, y_test)

@load joinpath(MODEL_DIR, "model.jld2") model

# ---------------------------------------------------------------------------------------------------
# Some example distributions
# ---------------------------------------------------------------------------------------------------

# Overwrite the function from gen.jl to deal with marginals constructed 
# as a sum of multiple variables, e.g. we wish to observe the total 
# of all full-length mRNA fragments (A + B + BC1 + ... + BC5 + C + D + E + F)
function ssa_extract_marg(sol_raw, marginal)
    map(2:size(sol_raw,2)) do i
        sol = @view sol_raw[marginal, i, :]
        sol = vec(sum(sol, dims=1))
        nmax = maximum(sol)
        hist = fit(Histogram, sol, 0:nmax+1, closed=:left)
        hist = normalize(hist, mode=:pdf)
        Float32.(hist.weights)
    end
end

# Initial conditions
ts = [100, 230, 360, 500, 620, 750, 880, 1000]
u0 = zeros(Int, numspecies(rn))
u0[1] = 1

# Set up the SSA simulations
jsys = convert(JumpSystem, rn, combinatoric_ratelaw=false)           
dprob = DiscreteProblem(jsys, u0, (0.0, last(ts)), zeros(Float64, numreactionparams(rn)))
jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))

# full-length mRNA (A + B + BC1 + ... + BC5 + C + D + E + F)
inds_FL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
# using 1M SSA trajectories to obtain smooth histograms
solver(ts, p) = ssa_solve(jprob, ts, p, 1000000; marginals=[inds_FL])

# rounded X_test[1043]
p = [0.009, 0.056, 1.22, 0.015, 0.81, 0.006, 0.009, 0.003, 0.78, 0.001, 0.057, 0.036, 0.017, 0.039, 0.069, 0.030, 0.003, 0.007]
p = Float32.(p)
t = 750f0

ps = vcat(t, p)
@time _data = build_dataset([t], [p], solver)
@save joinpath(MODEL_DIR, "example_SSA1.jld2") _data

#$750.0$ & $0.009$ & $0.056$ & $1.22$ & $0.015$ & $0.81$ & $0.006$ & $0.009$ & $0.003$ & $0.78$ & $0.001$ & $0.057$ & $0.036$ & $0.017$ & $0.039$ & $0.069$ & $0.03$ & $0.003$ & $0.007$

#@load joinpath(MODEL_DIR, "example_SSA1.jld2") _data
plt1 = plot_dist(_data[1][1], _data, model; true_label="SSA")
plt1 = plot!(plt1, xlabel="", ylabel="Probability", title="", leg=false,
             yticks=false, left_margin=4Plots.mm)

# rounded X_test[9362]
p = Float32.([0.016, 0.0214, 0.603, 0.0021, 2.338, 0.00037, 0.00191, 0.0163, 0.0128, 0.003, 0.0169, 0.0286, 0.0151, 0.062,
     0.0333, 0.0461, 0.0013, 0.0015])
t = 500f0

ps = vcat(t, p)
@time _data = build_dataset([t], [p], solver)
@save joinpath(MODEL_DIR, "example_SSA2.jld2") _data

#@load joinpath(MODEL_DIR, "example_SSA2.jld2") _data
plt2 = plot_dist(_data[1][1], _data, model; true_label="SSA")
plt2 = plot!(plt2, xlabel="FL mRNA number", ylabel="", title="", leg=false,
             yticks=false, left_margin=4Plots.mm)

#$500.0$ & $0.016$ & $0.0214$ & $0.603$ & $0.0021$ & $2.338$ & $0.00037$ & $0.00191$ & $0.0163$ & $0.0128$ & $0.003$ & $0.0169$ & $0.0286$ & $0.0151$ & $0.062$ & $0.0333$ & $0.0461$ & $0.0013$ & $0.0015$

l = @layout [a b]
plt = plot(plt1, plt2, layout=l, size=(330, 120), bottom_margin=0Plots.mm, top_margin=-2Plots.mm,
           guidefontsize=6, tickfontsize=6, legendfontsize=6, thickness_scaling=1.0, left_margin=1Plots.mm, right_margin=0Plots.mm,
           foreground_color_legend = nothing, fmt=:svg)

#savefig(joinpath(MODEL_DIR, "example_dists.svg"))

# ---------------------------------------------------------------------------------------------------
# Predicted vs true moments
# ---------------------------------------------------------------------------------------------------

# Plotting points only for t=1000
ind = 4
m = 4
m_NN = mean.(Distribution.(Ref(model), X_test[ind:m:end]))
var_NN = var.(Distribution.(Ref(model), X_test[ind:m:end]))

m_SSA = [sum((0:length(y)-1) .* y) for y in y_test[ind:m:end]]
var_SSA = [sum(((0:length(y)-1) .- m_SSA[i]) .^2  .* y) for (i, y) in enumerate(y_test[ind:m:end])];

scale = 1e-2
min_val = 0.0
max_val = maximum(vcat(m_SSA, m_NN))* 1.025 * scale
#xticks = (min_val:2:max_val, (min_val:2:max_val))
xticks = (0:2:max_val, (["0", (2:2:Int(ceil(max_val)))...]))
plt1 = plot(0:ceil(max_val), 0:ceil(max_val), linestyle=:dash, linecolor=:gray, linealpha=0.9, leg=false, 
             xlabel="Nessie", ylabel="SSA", title="Mean")
plt1 = scatter!(plt1, m_NN .* scale, m_SSA .*scale, grid = false, xlim=(min_val, max_val), ylim=(min_val, max_val), 
                markersize=2.5, markercolor=colorant"#0088c3ff", markerstrokecolor=colorant"#0088c3ff", 
                alpha=0.6, xticks=xticks, yticks=xticks)
plt1 = plot!(plt1, guidefontsize=6, tickfontsize=6, titlefontsize=6, thickness_scaling=1.0, 
             framestyle=:box, tick_direction=:out, right_margin=5Plots.mm)
plt1 = annotate!(plt1, [(-1.0, -1.0, Plots.text("×10²", 6, :black, :center))])

scale = 1e-4
min_val = 0.0
max_val = maximum(vcat(var_SSA, var_NN)) * 1.025 * scale
#xticks = (min_val:2:max_val, (min_val:2:max_val))
xticks = (0:2:max_val, (["0", (2:2:Int(ceil(max_val)))...]))
plt2 = plot(0:ceil(max_val), 0:ceil(max_val), linestyle=:dash, linecolor=:gray, linealpha=0.9, leg=false, 
            xlabel="Nessie", ylabel="", title="Variance")
plt2 = scatter!(plt2, var_NN .* scale, var_SSA .* scale, grid = false, xlim=(min_val, max_val), ylim=(min_val, max_val), 
                markersize=2.5, markercolor=colorant"#0088c3ff", markerstrokecolor=colorant"#0088c3ff", 
                alpha=0.6, xticks=xticks, yticks=xticks)
plt2 = plot!(plt2, guidefontsize=6, tickfontsize=6, titlefontsize=6, thickness_scaling=1.0,
             framestyle=:box, tick_direction=:out)
plt2 = annotate!(plt2, [(-1.5, -1.5, Plots.text("×10⁴", 6, :black, :center))])

plot(plt1, plt2, size=(260, 130), 
     left_margin=-1Plots.mm, bottom_margin=0Plots.mm, top_margin=-1Plots.mm, right_margin=0Plots.mm)

#savefig(joinpath(MODEL_DIR, "true_vs_predict_moments.svg"))