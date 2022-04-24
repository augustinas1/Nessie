using Catalyst

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const AFL_DIR = joinpath(DATA_DIR, "afl")

rn_afl = @reaction_network begin
      σ_u * (1 - G), 0 --> G + P
      σ_b, G + P --> 0
      ρ_u, G --> G + P
      ρ_b * (1 - G), 0 --> P
      1, P --> 0
end σ_u σ_b ρ_u ρ_b