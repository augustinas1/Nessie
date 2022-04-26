using Catalyst
using StaticArrays
using MAT
using DelaySSAToolkit
using LinearAlgebra

const DATA_DIR = joinpath(@__DIR__, "..", "data", "mapk")

# This file has to be obtained from the authors of reference [1] in the README.
data_tl = matread(joinpath(DATA_DIR, "DataTimeLapsed.mat"))
tt = data_tl["Data"]["Series"][4]["Cytometry"]["CytData"]["Time"][2:end]
cell_obs = map(x -> x[:], data_tl["Data"]["Series"][4]["Cytometry"]["pSTL1qV"]["Cells"][:])

const hog_tt = data_tl["Data"]["Series"][4]["Microscopy"]["Time"][:] ./ 60
const hog_data = data_tl["Data"]["Series"][4]["Microscopy"]["Hog"][:];

# The reaction pSTL_off --> pSTL_on has a time-dependent rate
# and is implemented separately using the DelaySSAToolkit
rn = @reaction_network begin
    60 * c2, pSTL_on --> pSTL_off
    (60 * c3 * 225, 60 * c4), pSTL_on <--> pSTLwCR
    60 * c5, pSTLwCR --> pSTLwCR + mRNA
    60 * c6, mRNA --> mRNA + pSTL1_qV
    60 * c7, pSTL1_qV --> 0
    60 * c8, mRNA --> 0
end hog1p_0 h Kd Vmax c2 c3 c4 c5 c6 c7 c8

## Time-dependent reaction
function draw_waiting_time(t, p)
    draw_waiting_time(Random.GLOBAL_RNG, t, p)
end

function get_c1(hog1p_s, p)
    hog1p_0, h, Kd, Vmax = @view p[1:4]
    hog1p = max(0, hog1p_s) + hog1p_0
    hog1p_h = hog1p^h
    60. * Vmax * (hog1p_h) / (Kd^h + hog1p_h)
end

# Draw the time of the next hog1 binding event (using the time-dependent hog1 signal)
function draw_waiting_time(rng, t, p)
    res = hog_tt[2]
    tidx = floor(Int, t / res)
    sum = 0.
    thresh = -log(rand(rng, Uniform()))
    
    coarseness = 3
    while tidx < length(hog_tt) + 1 - coarseness
        tidx += coarseness
        rate = get_c1(hog_data[tidx], p)
        sum += coarseness * res * rate
        
        if sum > thresh
            return max(res, hog_tt[tidx] - t)
        end
    end
    
    hog_tt[end] - t
end


delay_trigger_affect! = function (integrator, rng=missing)
    τ = draw_waiting_time(integrator.t, integrator.p)
    append!(integrator.de_chan[1], τ)
end

delay_trigger = Dict(1=>delay_trigger_affect!)
delay_complete = Dict(1=>[1=>1, 2=>-1])
delay_interrupt = Dict() 
delayjumpset = DelayJumpSet(delay_trigger, delay_complete, delay_interrupt)

## MAP parameters taken from Zechner et al.
gt_params = [ 1.581e-2, 6.130, 1.418e-1, 1.025,
              1.384, 6.669e-4, 1.469e-2, 2.825e-1,  5.663e-3, 5.476e-4, 1.283e-4 ]

u0 = @SVector [ 1, 0, 0, 0, 0 ]
tspan = (0., last(tt))

dprob = DiscreteProblem(rn, u0, tspan, gt_params)
jsys = convert(JumpSystem, rn)
djprob = DelayJumpProblem(jsys, dprob, DelayRejection(), delayjumpset, [[ ]], save_everystep=(false, false), save_positions=(false, false))

function ssa_extract_marg_dly(sol_raw, marginal)
    map(2:size(sol_raw)[2]) do i
        # SSA always saves t=0 values which we don't need
        sol = sol_raw[marginal, i, :]
        nmax = maximum(sol)
        hist = fit(Histogram, sol, 0:nmax+1, closed=:left)
        hist = normalize(hist, mode=:pdf)
        Float32.(hist.weights)
    end
end

function ssa_solve_dly(jprob, ts, p, n_traj; marginals)
    jprob = remake(jprob, tspan=(0., ts[end]), p=p)

    sol_SSA = solve(EnsembleProblem(jprob), SSAStepper(), saveat=ts, trajectories=n_traj)#, tstops=[0.01], callback=pcb)
    
    [ ssa_extract_marg_dly(sol_SSA, marg) for marg in marginals ]
end

solver(ts, p, N=1e4) = ssa_solve_dly(djprob, ts, p, N; marginals=[5])
    
logranges = [ -3 -1
               0 1.
              -2 0 
              -1 1 
              -1 1 
              -4 -2
              -3 -1
              -1 0 
              -3 -2
              -4 -3
              -5 -3
]

ranges = 10 .^ logranges
