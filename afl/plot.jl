using JLD2

include("../train_NN.jl")
include("../viz.jl")
include("../generate_data.jl")
include("afl.jl")

# load test data
@load joinpath(AFL_DIR, "test_data.jld2") X_test y_test
test_data = (X_test, y_test)

# load neural network
@load joinpath(AFL_DIR, "model.jld2") model

# FSP solver
fsp_sys = FSPSystem(rn_afl, combinatoric_ratelaw=false)
state_space = [2, 400]
u0 = zeros(state_space...)
u0[2, 1] = 1.0
fsp_prob = ODEProblem(fsp_sys, u0, (0., 1.), ones(numparams(rn_afl)))
solver(ts, p) = fsp_solve(fsp_prob, ts, p; marginals=[2], abstol=1e-6, reltol=1e-6)

# ---------------------------------------------------------------------------------------------------
# Some example distributions
# ---------------------------------------------------------------------------------------------------

#=
plt1 = plot_dist(X_test[78], test_data, model; true_label="SSA")
plt2 = plot_dist(X_test[75], test_data, model; true_label="FSP")
plt3 = plot_dist(X_test[92], test_data, model; true_label="FSP")
=#

p = Float32.([0.94, 0.01, 8.4, 28.1]); t = 10f0
ps = vcat(t, p); _data = build_dataset([t], [p], solver)
plt1 = plot_dist(ps, _data, model; true_label="FSP")
plt1 = plot!(plt1, xlabel="", ylabel="Probability", title="", yticks=false, left_margin=4Plots.mm, leg=true)

p = Float32.([0.69, 0.07, 7.2, 40.6]); t = 25f0
ps = vcat(t, p); _data = build_dataset([t], [p], solver)
plt2 = plot_dist(ps, _data, model)
plt2 = plot!(plt2, xlabel="Protein number", ylabel="", title="", yticks=false, leg=false, xticks=([0, 20, 40, 60], 0:20:60))

p = Float32.([0.44, 0.08, 0.94, 53.13]); t = 100f0
ps = vcat(t, p); _data = build_dataset([t], [p], solver)
plt3 = plot_dist(ps, _data, model)
plt3 = plot!(plt3, xlabel="", ylabel="", title="", yticks=false,leg=false, right_margin=-1Plots.mm)

plt = plot(plt1, plt2, plt3, layout=(1,3), size=(350, 110), bottom_margin=0Plots.mm, top_margin=-2Plots.mm,
           right_margin=-1Plots.mm, left_margin=1Plots.mm, guidefontsize=6, tickfontsize=6, legendfontsize=6, thickness_scaling=1.0,
           foreground_color_legend = nothing, fmt=:svg)
#savefig(joinpath(AFL_DIR, "example_dists.svg"))

# ---------------------------------------------------------------------------------------------------
# Bimodality heatmaps
# ---------------------------------------------------------------------------------------------------

using LaTeXStrings

cmap_new = [ colorant"#0d89c4", colorant"#2996cb", colorant"#46a4d2", colorant"#63b2d9", colorant"#80c0e0", 
             colorant"#9dcee7", colorant"#badcee", colorant"#d7eaf5", colorant"#f4f8fc", colorant"#eae0e5", 
             colorant"#e1c8ce", colorant"#d7b0b7", colorant"#ce99a0", colorant"#c58189", colorant"#bb6972", 
             colorant"#b2515b", colorant"#a93a45" ]

function plot_bimodality_NN(model, ind1, ind2, iter1, iter2, sym1, sym2, ps, t)

    bcs = Matrix{Float64}(undef, length(iter1), length(iter2))
    ps_all = vcat(t, ps...) 
    ind1 += 1; ind2 +=1

    for (i, p1) in enumerate(iter1)
        
        for (j, p2) in enumerate(iter2)
            
            ps_all[ind1] = p1; ps_all[ind2] = p2 
            mnb = Distribution(model, ps_all)
            bcs[i, j] = bimodcoeff(mnb)
            
        end
        
    end

    plt = contourf(iter1, iter2, bcs', linewidth=0.005, c=cmap_new, linecolor=:black)
    plt = plot!(plt, xlabel=sym1, ylabel=sym2)
    plt, bcs

end

function plot_bimodality_FSP(ind1, ind2, iter1, iter2, sym1, sym2, ps, t)

    fsp_sys = FSPSystem(rn_afl, combinatoric_ratelaw=false)
    state_space = [2, 400]
    u0 = zeros(state_space...)
    u0[2, 1] = 1.0
    fsp_prob = ODEProblem(fsp_sys, u0, (0., t), ones(numparams(rn_afl)))

    bcs = Matrix{Float64}(undef, length(iter1), length(iter2))

    Threads.@threads for (i, j) in collect(Iterators.product(1:length(iter1), 1:length(iter2)))
        p1 = iter1[i]
        p2 = iter2[j]

        ps_ = copy(ps)
        ps_[ind1] = p1; ps_[ind2] = p2 
        sol_raw = solve(fsp_prob, CVODE_BDF(), saveat=[t], p=ps_; reltol=1e-8, abstol=1e-8)

        dist = sum(sol_raw.u[1], dims=1)[1:end]
        max_ind = maximum(findall(val -> !isapprox(val, 0f0, atol=1e-5), dist))
        dist = dist[1:max_ind]

        m = sum( (0:max_ind-1) .* dist )
        s = sqrt(sum( ((0:max_ind-1) .- m) .^2 .* dist ))
        m3 = sum(((0:max_ind-1) .- m) .^3 .* dist)
        m4 = sum( ((0:max_ind-1) .- m) .^4 .* dist)
        bcs[i, j] = 1 / (m4/s^4 - (m3 / s^3)^2)
        
    end

    plt = contourf(iter1, iter2, bcs', linewidth=0.02, c=cmap_new)
    plt = plot!(plt, xlabel=sym1, ylabel=sym2)

    plt, bcs

end

# σ_u σ_b ρ_u ρ_b
ranges = [ 0 2
           0 0.1
           0 10
           0 100 ]

ps = [1f0, 0.05f0, 1f0, 20f0]

iter1 = 0.01:0.002:0.1
iter2 = 1:1:100
sym1 = "σ_b"
sym2 = "ρ_b"
@time plt1, bcs_nn = plot_bimodality_NN(model, 2, 4, iter1, iter2, sym1, sym2, ps, 100f0)
plt1 = plot!(plt1, xlabel=L"\sigma_b", ylabel=L"\rho_b", title="Nessie", xticks=xticks)

#@time plt2, bcs_fsp = plot_bimodality_FSP(2, 4, iter1, iter2, sym1, sym2, ps, 100f0)
#plt2 = plot!(plt2, xlabel=L"\sigma_b", ylabel=L"\rho_b"); plot(plt2)
#@save joinpath(AFL_DIR, "heatmap_bcs_FSP.jld2") bcs_fsp

xticks = (0.025:0.025:0.1, 0.025:0.025:0.1)
@load joinpath(AFL_DIR, "heatmap_bcs_FSP.jld2") bcs_fsp
plt2 = contourf(iter1, iter2, bcs_fsp', c=cmap_new, linewidth=0.005, linecolor=:black, xticks=xticks)
plt2 = plot!(plt2, xlabel=L"\sigma_b", ylabel=L"\rho_b", title="FSP")

lim1, lim2 = extrema(hcat(bcs_nn, bcs_fsp))
plt = plot(plt1, plt2, clim=(lim1, lim2), size = (370, 130), tick_direction = :out, thickness_scaling=1.0,
           guidefontsize=6, tickfontsize=6, legendfontsize=6, ticksize=6, titlefontsize=6,
           left_margin=-1Plots.mm, bottom_margin=0.5Plots.mm, top_margin=-1Plots.mm, right_margin=-1Plots.mm, 
           framestyle=:box)
#savefig(joinpath(AFL_DIR, "bimodality.svg"))

# ---------------------------------------------------------------------------------------------------
# Predicted vs true moments
# ---------------------------------------------------------------------------------------------------

# considering only t=100
m_NN = mean.(Distribution.(Ref(model), X_test[1:4:end]))
var_NN = var.(Distribution.(Ref(model), X_test[1:4:end]))

m_FSP = [sum((0:length(y)-1) .* y) for y in y_test[1:4:end]]
var_FSP = [sum(((0:length(y)-1) .- m_FSP[i]) .^2  .* y) for (i, y) in enumerate(y_test[1:4:end])] 

max_val = maximum(vcat(m_FSP, m_NN))*1.025
plt1 = plot(1:max_val, 1:max_val, linestyle=:dash, linecolor=:gray, linealpha=0.9, legend=false, xlabel="Nessie", ylabel="FSP", title="Mean")
plt1 = scatter!(plt1, m_NN, m_FSP, color=colorant"#0088c3ff", grid = false, xlim=(0, max_val), ylim=(0, max_val), markersize=2.5, markerstrokecolor=colorant"#0088c3ff", alpha=0.6)
plt1 = plot!(plt1, guidefontsize=6, tickfontsize=6, titlefontsize=6, thickness_scaling=1.0, framestyle=:box, tick_direction=:out)
#savefig(joinpath(AFL_DIR, "true_vs_predict_mean.svg"))

max_val = maximum(vcat(var_FSP, var_NN))*1.025
plt2 = plot(1:max_val, 1:max_val, linestyle=:dash, linecolor=:gray, linealpha=0.9, legend=false, xlabel="Nessie", title="Variance")
plt2 = scatter!(plt2, var_NN, var_FSP, color=colorant"#0088c3ff", grid = false, xlim=(0, max_val), ylim=(0, max_val), markersize=2.5, markerstrokecolor=colorant"#0088c3ff", alpha=0.6)
plt2 = plot!(plt2, guidefontsize=6, tickfontsize=6, titlefontsize=6, thickness_scaling=1.0, framestyle=:box, tick_direction=:out)
#savefig(joinpath(AFL_DIR, "true_vs_predict_variance.svg"))

plt = plot(plt1, plt2, size=(290, 130), left_margin=-1Plots.mm, bottom_margin=0Plots.mm, top_margin=-1Plots.mm, right_margin=1Plots.mm)
#savefig(joinpath(AFL_DIR, "true_vs_predicted_moments.svg"))