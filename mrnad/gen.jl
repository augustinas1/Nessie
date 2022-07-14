# Generate training, validation and test set data for the model of mRNA turnover
using Sobol
using JLD2

include("../generate_data.jl")
include("model.jl")

# Create initial conditions
ts = [100, 230, 360, 500, 620, 750, 880, 1000]
u0 = zeros(Int, numspecies(rn))
u0[1] = 1

ranges = [ 0.0001 0.1   # σ_u
           0.0001 0.1   # σ_b
           0.1    2.0   # k_2
           0.001  0.02  # k_3
           0.5    4     # k_4
           0.0001 0.01  # k_5
           0.001  0.02  # k_8
           0.001  0.02  # k_9
           0.01   0.2   # k_10
           0.001  0.01  # k_11
           0.01   0.1   # r_1
           0.01   0.1   # r_2
           0.01   0.1   # r_3
           0.01   0.1   # r_4
           0.01   0.1   # r_5
           0.01   0.05  # r_6
           0.001  0.01  # r_7 
           0.001  0.01  # r_8
         ]

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

# Convert reaction network into a JumpProblem for use with the SSA
jsys = convert(JumpSystem, rn, combinatoric_ratelaw=false)           
dprob = DiscreteProblem(jsys, u0, (0.0, last(ts)), zeros(Float64, 18))
jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))

# Full-length mRNA (A + B + BC1 + ... + BC5 + C + D + E + F)
inds_FL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]
solver(ts, p) = ssa_solve(jprob, ts, p, 1000; marginals=[inds_FL])
# We need more SSA samples for the validation and test sets to get an accurate comparison
solver_accurate(ts, p) = ssa_solve(jprob, ts, p, 100000; marginals=[inds_FL])

# Sampling parameters using a logarithmic Sobol sequence
d = size(ranges, 1)
seq = LogSobolSeq(ranges[:,1], ranges[:,2])

@time train_pts = [ Sobol.next!(seq) for i in 1:100000 ]
@time valid_pts = [ Sobol.next!(seq) for i in 1:100 ]
@time test_pts = [ Sobol.next!(seq) for i in 1:1000 ]

X_test, y_test = build_dataset(ts, test_pts, solver_accurate)
@save joinpath(MODEL_DIR, "test_data.jld2") X_test y_test
X_train, y_train = build_dataset(ts, train_pts, solver)
@save joinpath(MODEL_DIR, "train_data.jld2") X_train y_train
X_valid, y_valid = build_dataset(ts, valid_pts, solver_accurate)
@save joinpath(MODEL_DIR, "valid_data.jld2") X_valid y_valid
