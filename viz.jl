using Plots

@userplot SSADist

@recipe function f(dist::SSADist; true_label="SSA")
    yy = dist.args[1]
    push!(yy, 0f0)
    nmax = length(yy) - 1

    @series begin
        seriestype := :steppost
        seriescolor --> colorant"#e4f0f8ff"
        label --> true_label
        
        linecolor --> nothing
        fillrange --> yy
        (0:nmax) .- 0.5, zeros(nmax+1)
    end

    @series begin
        seriestype := :steppost

        linecolor --> colorant"#808080ff"
        linewidth --> 0.5
        linealpha --> 0.8
        
        label --> ""
        
        # +1 to finish the contour nicely
        (0:nmax) .- 0.5, yy
    end
    
end

@userplot CMENetDist

@recipe function f(dist::CMENetDist; nmax=nothing)
    loc = dist.args[1]
    model = dist.args[2]

    mnb = Distribution(model, loc)

    if nmax === nothing
        nmax = ceil(Int, mean(mnb) + 3 * std(mnb))
    end
    yy = pred_pdf(model, loc, 0:nmax)
    
    @series begin
        seriestype := :steppost

        linecolor --> colorant"#0088c3ff"
        linealpha --> 0.9
        linewidth --> 1.5

        label --> "Nessie"

        (0:nmax) .- 0.5, yy
    end
end

plot_dist(args...; kwargs...) = plot_dist!(plot(), args...; kwargs...)

function plot_dist!(plt::AbstractPlot, loc, data, model; params=nothing, true_label="SSA", kwargs...)
    X, y = data
   
    ind = findfirst(x -> x == loc, X)

    ind === nothing && error("Could not find parameters in data: $loc")

    ssadist!(plt, y[ind]; true_label)
    cmenetdist!(loc, model; nmax=length(y[ind]) - 1)

    title = if params === nothing
        "t = $(loc[1])"
    else
        "t = $(loc[1]), " * join([ params[i] = loc[i+1] for i in 1:length(loc) - 1 ], ", ")
    end

    ylim = maximum(Plots.axis_limits(plt.subplots[1], :y))
    plot!(plt, xlabel="n", ylabel="P(n)", grid=nothing,
          xlims=(-0.5, Inf), ylims=(0, ylim), tick_direction=:out,
          title=title; kwargs...)
end


function loss_heatmap(data, model, ps; loss = loss_crossentropy, syms=nothing, kwargs...)

	X, y = data
	p_inds = findall(p -> !isnothing(p), ps)
	inds = findall(col -> col[p_inds] == ps[p_inds], X)

	X = @view X[inds]
	y = @view y[inds]
	ls = loss.(X, y, Ref(model))

	inds = setdiff(1:length(ps), p_inds)
	max1 = maximum(x[inds[1]] for x in X)
	dim1 = findfirst(x -> x[inds[1]] == max1, X)
	v1 = [x[inds[1]] for x in X[1:dim1]]
	dim2 = Int(length(y) / dim1)
	v2 = [x[inds[2]] for x in X[1:dim1:length(y)]]

	plt = heatmap(v1, v2, reshape(ls, (dim1, dim2))')
	if !isnothing(syms)
		plot!(plt, xlabel=syms[inds[1]], ylabel=syms[inds[2]])
		plot!(plt, title=join(("$(syms[i]) = $(ps[i])" for i in p_inds), ", "))
	end
	plot!(plt; kwargs...)

end


## Interactive plotting utilities

function plot_losses(clear=true)
    train_losses = Float32[]
    val_losses = Float32[]
    rounds = Int[]
    
    ax = plot(Int[], Float32[], yscale=:log10, label="train", color=:blue)
    plot!(ax, Int[], Float32[], label="valid", color=:orange)
    eta = 0f0
    
    rounds = ax.series_list[1].plotattributes[:x]
    rounds_val = ax.series_list[2].plotattributes[:x]
    
    train_losses = ax.series_list[1].plotattributes[:y]
    val_losses = ax.series_list[2].plotattributes[:y]
    
    ymax = -Inf
    ymin = Inf
        
    annotate!(ax, 0, 0, text("0", :blue, :right, 8))
    annotate!(ax, 0.1, 0, text("0", :orange, :right, 8))
    
    plot!(ax, size=(500,250))
    
    return (eta_new, train_loss, val_loss) -> begin
        rd = length(rounds) + 1
        push!(rounds, rd)
        push!(rounds_val, rd)
        
        push!(train_losses, train_loss)
        push!(val_losses, val_loss)
        
        if eta_new != eta
            eta = eta_new
            vline!(ax, [rd], linestyle=:dash, color=:red, label=false)
        end
        
        ymax = max(ymax, val_loss, train_loss)
        ymin = min(ymin, val_loss, train_loss)
        
        plot!(ax, xlims=(1,rd*1.1), ylims=(ymin, ymax), show=true)
        
        ax.subplots[1].attr[:annotations][1] = (rd, train_loss, text("$train_loss", :blue, :left, 8))
        ax.subplots[1].attr[:annotations][2] = (rd, val_loss, text("$val_loss", :orange, :left, 8))
        
        clear && IJulia.clear_output(true)
        display(ax)
    end
end

function plot_sample_dists(test_data, model, clear=true)
    X_test = first(test_data)
    
    return (eta_new, train_loss, val_loss) -> begin    
        plts = [ plot_dist(X_test[i], test_data, model, legend=(i == 1)) for i in 1:length(X_test) ]
        
        clear && IJulia.clear_output(true)
        display(plot(plts..., size=(500, 500)))
    end
end

function combine(fcts...)
    return (args...) -> foreach(f -> f(args...), fcts)
end