@testset "intervention" begin
    h = Hard(1)
    @test h.a_star == 1
    s = Soft(1, 0.1)
    @test s.a_star == 1 && s.ε == 0.1
    @test HCM.default_baseline(Hard(1)) == Hard(0)
    @test HCM.default_baseline(Soft(1, 0.2)) == Soft(1, 0.0)
end
