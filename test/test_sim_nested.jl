@testset "sim nested" begin
    out = sim_hcm(:nested_confounder; n=20, m=8,
                  params=(classes_per_school=4, β1=0.5, sd_school=1.0, sd_class=0.7,
                          ρ_school=1.0, ρ_class=1.0, sd_y=0.3), family=:gaussian, seed=1)
    @test Set(names(out.data)) == Set(["school","class","student","a","y"])
    @test out.true_hard ≈ 0.5
    @test out.true_soft(1, 0.2) ≈ 0.2 * mean(0.5 .* (1 .- out.p_class))
    @test length(out.p_class) == 20*4
end
