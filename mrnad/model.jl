using Catalyst

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const MODEL_DIR = joinpath(DATA_DIR, "mrnad")

rn = @reaction_network begin
    σ_u*(1-Gene), 0 --> Gene
    σ_b, Gene --> 0
    1, Gene --> Gene + A
    k_2, A --> B
    r_1, B --> BC1
    r_2, BC1 --> BC2
    r_3, BC2 --> BC3
    r_4, BC3 --> BC4
    r_5, BC4 --> BC5
    r_6, BC5 --> C
    k_3, C --> D
    r_7, C --> E
    k_4, D --> L
    r_7, D --> F
    k_9, E --> F
    k_8, E --> G
    r_8, L --> I2
    k_4, F --> I1
    k_8, F --> M
    k_10, G --> M
    k_11, G --> 0
    k_5, I2 --> 0
    k_5, I1 --> 0
    k_4, M --> 0
    k_11, M --> 0
end σ_u σ_b k_2 k_3 k_4 k_5 k_8 k_9 k_10 k_11 r_1 r_2 r_3 r_4 r_5 r_6 r_7 r_8