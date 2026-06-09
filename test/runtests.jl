using HCM, Test, Statistics, StatsFuns, DataFrames

@testset "HCM.jl" begin
    # test files are included as they are implemented
    include("test_intervention.jl")
    include("test_estimand.jl")
    include("test_sim.jl")
end
