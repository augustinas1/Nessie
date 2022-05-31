using FiniteStateProjection
using OrdinaryDiffEq
using Sundials
using DiffEqJump
using Distributions
using Sobol
using StatsBase, LinearAlgebra

using ProgressMeter
ProgressMeter.ijulia_behavior(:append)
ProgressMeter.ijulia_behavior(:clear)

##

"""
Wrapper for logarithmically spaced Sobol sequences. Implements the standard
SobolSeq interface, except the points sampled will be approximately uniformly
distributed when viewed in log-space.
"""
struct LogSobolSeq{N,T} <: Sobol.AbstractSobolSeq{N}
    seq::ScaledSobolSeq{N,T}
end

Sobol.next!(s::LogSobolSeq, x::AbstractVector{<:AbstractFloat}) = 10 .^ Sobol.next!(s.seq, x)
Sobol.next!(s::LogSobolSeq) = 10 .^ Sobol.next!(s.seq)

function Sobol.skip!(s::LogSobolSeq, n::Integer, x; exact=false)
    skip!(s.seq, n, x; exact=exact)
    s
end

Base.skip(s::LogSobolSeq, n::Integer; exact=false) = Sobol.skip!(s, n, Array{Float64,1}(undef, ndims(s)); exact=exact)

function LogSobolSeq(N::Integer, lb, ub)
    LogSobolSeq(SobolSeq(log10.(lb), log10.(ub)))
end

LogSobolSeq(lb, ub) = LogSobolSeq(length(lb), lb, ub)

##

"""
	build_dataset(ts, ps, solver)

	Generates a training, validation or test dataset at parameters `ps` and times
	`ts`. The output is a tuple `(X, y)`, where each entry of `X` consists of an
	input to the neural network of the form `[ t, params... ]` and the corresponding
	entry of `y` consists of the training data at that point in the form of a histogram.

	The training data is generated using the callback `solver(ts, p)`, which takes a vector of 
	times `ts` and a single parameter set `p`, and returns a vector of histograms, one for each
	time point in `ts`. The standard choices are `ssa_solve` and `fsp_solve`, detailed below.
"""
function build_dataset(ts, ps, solver)
    progress = Progress(length(ps), 1, "Generating data... ")

    X = Vector{Float32}[]
    y = Vector{Float32}[]
    
    for p in ps
        ProgressMeter.next!(progress; showvalues = [(:parameter_set, p)])
        
        ret = solver(ts, p)

        for ret_i in ret
            for (t, ret_it) in zip(ts, ret_i)
                push!(X, [t, p...])
                push!(y, ret_it)
            end
        end
    end

    X, y
end


"""
	build_dataset_parallel(ts, ps, solver)

	As `build_dataset`, but uses multithreading to run the solver at multiple parameter
	sets in parallel. If training data are generated using the SSA (via `ssa_solve`), then
	each call to `solver` already uses multithreading internally andthis function will likely 
	offer no advantages over `build_dataset`. 
"""
function build_dataset_parallel(ts, ps, solver)
    #progress = Progress(length(ps), 1, "Generating data... ")

    X = Vector{Vector{Vector{Float32}}}()
    y = Vector{Vector{Vector{Float32}}}()
    
    for i in 1:Threads.nthreads()
        push!(X, Float32[])
        push!(y, Float32[])
    end
    
    Threads.@threads for p in ps
        #ProgressMeter.next!(progress; showvalues = [(:parameter_set, p)])
        
        ret = solver(ts, p)

        for ret_i in ret
            for (t, ret_it) in zip(ts, ret_i)
                push!(X[Threads.threadid()], [t, p...])
                push!(y[Threads.threadid()], ret_it)
            end
        end
    end

    vcat(X...), vcat(y...)
end

##

"""
	ssa_solve(jprob, ts, p, n_traj; marginals)

	Runs `n_traj` simulations of the Jump Problem `jprob` with parameters `p`
	and returns the results at times `ts` for the marginals `marginals`. The result
	is a vector of vectors such that `ret[m][i]` is the histogram for the `m`-th 
	specified marginal at the `i`-th time point.

	This function uses multithreading if possible via the `EnsembleProblem` interface.

	IMPORTANT: `jprob` must have `save_positions = (false, false)` for this function to
	work correctly.
"""
function ssa_solve(jprob, ts, p, n_traj; marginals)
    jprob = remake(jprob, tspan=(0., ts[end]), p=p)

    sol_SSA = solve(EnsembleProblem(jprob), SSAStepper(), saveat=ts, trajectories=n_traj)
    
    [ ssa_extract_marg(sol_SSA, marg) for marg in marginals ]
end

"""
	Extract marginal distributions from an EnsembleSolution at fixed time points, which
	are those at which the solutions were saved (set by the `saveat` solver option).
"""
function ssa_extract_marg(sol_raw, marginal)
    map(2:size(sol_raw,2)) do i
        # SSA always saves t=0 values which we don't need
        sol = @view sol_raw[marginal, i, :]
        nmax = maximum(sol)
        hist = fit(Histogram, sol, 0:nmax+1, closed=:left)
        hist = normalize(hist, mode=:pdf)
        Float32.(hist.weights)
    end
end

##

""" 
	Analogous to `ssa_solve`, but takes an ODEProblem and uses the FSP to compute the
 	target distributions.
"""
function fsp_solve(fsp_prob, ts, p; marginals, solver=CVODE_BDF(), kwargs...)
    fsp_prob = remake(fsp_prob, p=p, tspan=(0., ts[end]))
    sol_raw = solve(fsp_prob, solver, saveat=ts; kwargs...)

    fsp_extract_margs(sol_raw, marginals)
end


function fsp_extract_margs(sol_raw, marginals)
    n_species = ndims(sol_raw) - 1
    dropped_dims = Tuple(setdiff(1:n_species, marginals))
    sol_FSP = dropdims(sum(sol_raw, dims=dropped_dims), dims=dropped_dims)

    [ fsp_extract_marg(sol_FSP, i) for i in 1:length(marginals) ]
end

# Use abs to prevent negative values
function fsp_extract_marg(sol_FSP, marginal::Int)
    n_species = ndims(sol_FSP) - 1
    dropped_dims = Tuple(setdiff(1:n_species, marginal))
    sol_marg = dropdims(sum(sol_FSP, dims=dropped_dims), dims=dropped_dims)

    n_ts = size(sol_marg, 2)
    map(1:n_ts) do i
        sol = @view sol_marg[:, i]
        max_ind = maximum(findall(val -> !isapprox(val, 0f0, atol=1e-5), sol))
        abs.(Float32.(@view sol[1:max_ind]))
    end
end

##
