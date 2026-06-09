using HCM, Test, Statistics, StatsFuns

@testset "HCM.jl" begin
    # test files are included as they are implemented
    include("test_intervention.jl")
    include("test_estimand.jl")
end
