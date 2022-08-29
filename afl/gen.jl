# Generate training, validation and test data for the AFL example
using DiffEqBase, Sobol, JLD2

include("afl.jl")
include("../generate_data.jl")

# We use the FSP instead of the SSA in this demonstrative example
fsp_sys = FSPSystem(rn_afl, combinatoric_ratelaw=false)
u0 = zeros(2, 400)
u0[2, 1] = 1.0
fsp_prob = convert(ODEProblem, fsp_sys, u0, (0., 1.), ones(numparams(rn_afl)))
solver(ts, p) = fsp_solve(fsp_prob, ts, p; marginals=[2], abstol=1e-6, reltol=1e-6)

# time snapshots
ts = [5, 10, 25, 100]

# ranges for afl parameters
# σ_u σ_b ρ_u ρ_b
ranges = [ 0 2
           0 0.1
           0 10
           0 100 ]

# Draw training, validation and test parameters from a Sobol sequence
s = SobolSeq(ranges[:,1], ranges[:,2])

ps_train = [ Sobol.next!(s) for i in 1:1000 ]
ps_valid = [ Sobol.next!(s) for i in 1:100 ]
ps_test = [ Sobol.next!(s) for i in 1:500 ]

# Use the FSP to create training dataset
X_train, y_train = build_dataset_parallel(ts, ps_train, solver)
@save joinpath(AFL_DIR, "train_data.jld2") X_train y_train

X_valid, y_valid = build_dataset_parallel(ts, ps_valid, solver)
@save joinpath(AFL_DIR, "valid_data.jld2") X_valid y_valid

X_test, y_test = build_dataset_parallel(ts, ps_test, solver)
@save joinpath(AFL_DIR, "test_data.jld2") X_test y_test
