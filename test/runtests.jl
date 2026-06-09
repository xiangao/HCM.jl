using HCM, Test, Statistics, StatsFuns, DataFrames, StatsModels

@testset "HCM.jl" begin
    # test files are included as they are implemented
    include("test_intervention.jl")
    include("test_estimand.jl")
    include("test_sim.jl")
    include("test_sim_nested.jl")
    include("test_spec.jl")
    include("test_spec_nested.jl")
    include("test_nested_estimand.jl")
    include("test_confounder.jl")
    include("test_methods.jl")
    include("test_compare_fe.jl")
    include("test_nested.jl")
end
