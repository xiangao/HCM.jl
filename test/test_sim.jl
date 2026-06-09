@testset "sim" begin
    out = sim_hcm(:confounder; n=40, m=30,
                  params=(β0=0.0, β1=0.5, sd_u=1.0, α_p=0.0, ρ=1.0, sd_y=0.3),
                  family=:gaussian, seed=1)
    @test Set(names(out.data)) == Set(["unit","subunit","a","y"])
    @test nrow(out.data) == 40*30
    @test out.true_hard ≈ 0.5
    @test out.true_soft(1, 0.2) ≈ 0.2 * mean(0.5 .* (1 .- out.p))
    @test length(out.u) == 40 && length(out.p) == 40
    @test 0 < mean(out.data.a) < 1
end
