module HCM
using Distributions, DataFrames, StatsModels, StatsFuns, Statistics, Random
using Turing, MCMCChains, FixedEffectModels

include("intervention.jl")
export Hard, Soft

include("estimand.jl")
export ate, link_inv

include("sim.jl")
export sim_hcm
end # module
