using Sobol
using JLD2
using Catalyst
using Random

include("../generate_data.jl")
include("network.jl")

solver_train(ts, p) = ssa_solve_dly(djprob, ts, p, 1000; marginals=[5])
solver_acc(ts, p) = ssa_solve_dly(djprob, ts, p, 100000; marginals=[5])

seq = LogSobolSeq(ranges[:,1], ranges[:,2])

test_pts = [ Sobol.next!(seq) for i in 1:100 ]
val_pts = [ Sobol.next!(seq) for i in 1:250 ]
train_pts = [ Sobol.next!(seq) for i in 1:15000 ];

X_train, y_train = build_dataset(tt, train_pts, solver_train)
@save joinpath(DATA_DIR, "train_data.jld2") X_train y_train

X_test, y_test = build_dataset(tt, test_pts, solver_acc)
@save joinpath(DATA_DIR, "test_data.jld2") X_test y_test

X_val, y_val = build_dataset(tt, val_pts, solver_acc)
@save joinpath(DATA_DIR, "val_data.jld2") X_val y_val
