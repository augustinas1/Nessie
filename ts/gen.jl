""" Generate training, validation and test set data for the toggle switch example """
using Sobol
using JLD2

include("../generate_data.jl")
include("model.jl")

ts_train = 2:2:100
ts_valid = 1:1:100

# Create initial conditions
@variables t G_uA(t) G_uB(t) P_A(t) P_B(t)
u0 = zeros(Int, numspecies(rn))
u0[[speciesmap(rn2)[i] for i in (G_uA, G_uB)]] .= 1

ranges = [ 0   0.0005  # σ_bB 
           0   0.1     # σ_uB
           0   0.0005  # σ_bA
           0   0.1     # σ_uA
           0   500     # ρ_uA
           0   500     # ρ_bA
           0   12      # γ_A 
           1   20      # δ_mA  
           0   2       # δ_p
           0   500     # ρ_uB
           0   500     # ρ_bB 
           0   12      # γ_B
           1   20      # δ_mB
           0   2       # σ_uM
           0   0.5     # σ_bM
           0   100     # δ_pm   
         ]

# Convert reaction network into a JumpProblem for use with the SSA
jsys = convert(JumpSystem, rn, combinatoric_ratelaw=false)           
dprob = DiscreteProblem(jsys, u0, (0.0, last(ts_train)), zeros(Float64, numreactionparams(rn)))
jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))

idx = speciesmap(rn)[P_B]   # target protein
# We need more SSA samples for the test set to get an accurate comparison
solver_accurate(ts, p) = ssa_solve(jprob, ts, p, 100000; marginals=[idx])
solver(ts, p) = ssa_solve(jprob, ts, p, 1000; marginals=[idx])

d = size(ranges, 1)
seq = SobolSeq(ranges[:,1], ranges[:,2])

# Warning: the following can take a long time to run!
# this was initially done by piecing together a number of different smaller runs
@time train_pts = [ Sobol.next!(seq) for i in 1:40000 ]
@time valid_pts = [ Sobol.next!(seq) for i in 1:100 ]
@time test_pts = [ Sobol.next!(seq) for i in 1:500 ]

#@time X_train, y_train = build_dataset(ts_train, train_pts, solver)
#@time X_valid, y_valid = build_dataset(ts_valid, valid_pts, solver_accurate)
#@time X_test, y_test = build_dataset(ts_valid, test_pts, solver_accurate)

#@save joinpath(MODEL_DIR, "train_data.jld2") X_train y_train
#@save joinpath(MODEL_DIR, "valid_data.jld2") X_valid y_valid
#@save joinpath(MODEL_DIR, "test_data.jld2") X_test y_test
