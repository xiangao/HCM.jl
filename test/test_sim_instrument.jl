@testset "sim instrument" begin
    out = sim_hcm(:instrument; n=40, m=25,
                  params=(θ0=0.0, θa=0.5, θr0=1.0, θr1=-0.5, sd_u=1.0, α0=0.0, ρ=1.0, β0=1.0,
                          sd_ω=1.0, sd_y=0.3), family=:gaussian, seed=1)
    @test Set(names(out.data)) == Set(["unit","subunit","z","a","y"])
    @test out.true_hard ≈ 0.5
    @test out.true_soft(1, 0.2) ≈ 0.5 * 0.2 * mean(1 .- out.qa)
    @test length(out.qa) == 40 && length(out.ω) == 40
    @test all(i -> length(unique(out.data.y[out.data.unit .== i]))==1, 1:40)
    @test length(unique(round.(out.ω, digits=6))) > 1
end
