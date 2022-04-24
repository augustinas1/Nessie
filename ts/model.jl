using Catalyst

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const MODEL_DIR = joinpath(DATA_DIR, "ts")

rn = @reaction_network begin
    (σ_bB, σ_uB), P_A + G_uB <--> G_bB
    (σ_bA, σ_uA), P_B + G_uA <--> G_bA
    ρ_uA, G_uA --> G_uA + M_A
    ρ_bA, G_bA --> G_bA + M_A
    γ_A, M_A --> M_A + P_A
    δ_mA, M_A --> 0
    δ_p, P_A --> 0
    ρ_uB, G_uB --> G_uB + M_B
    ρ_bB, G_bB --> G_bB + M_B
    γ_B, M_B --> M_B + P_B
    δ_mB, M_B --> 0
    1, P_B --> 0
    (σ_bM, σ_uM), P_A + M_B <--> PAMB
    δ_pm, PAMB --> 0
end σ_bB σ_uB σ_bA σ_uA ρ_uA ρ_bA γ_A δ_mA δ_p ρ_uB ρ_bB γ_B δ_mB σ_uM σ_bM δ_pm

#= notation more consistent with Thomas et al. (2014)
rn = @reaction_network begin
    (a_10, a_01), P_1 + G_2 <--> G2P1  # a_10 / Ω
    (b_10, b_01), P_2 + G_1 <--> G1P2  # b_10 / Ω
    a_2, G_1 --> G_1 + M_1             # a_2*Ω
    a_3, G1P2 --> G1P2 + M_1           # a_3*Ω
    a_4, M_1 --> M_1 + P_1
    a_5, M_1 --> 0
    a_6, P_1 --> 0
    b_2, G_2 --> G_2 + M_2             # b_2*Ω
    b_3, G2P1 --> G2P1 + M_2           # b_3*Ω
    b_4, M_2 --> M_2 + P_2
    b_5, M_2 --> 0                         
    1, P_2 --> 0                    
    (b_8, b_7), P_1 + M_2 <--> M2P1    # b_8 / Ω
    b_9, M2P1 --> 0 
end a_10 a_01 b_10 b_01 a_2 a_3 a_4 a_5 a_6 b_2 b_3 b_4 b_5 b_7 b_8 b_9
=#