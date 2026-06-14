@testset "nested recovery" begin
    if get(ENV, "HCM_SAMPLING_TESTS", "") != "1"
        @test_skip "set HCM_SAMPLING_TESTS=1 to run NUTS recovery tests"
    else
        sim = sim_hcm(:nested_confounder; n=25, m=10, params=(classes_per_school=4, β1=0.5,
                      ρ_school=1.0, ρ_class=1.0, sd_y=0.3), family=:gaussian, seed=11)
        fit = hcm(@formula(y ~ a), sim.data; groups=(:school,:class), motif=:nested_confounder,
                  iter=500, chains=2)
        ah = ate(fit, Hard(1); baseline=Hard(0)); q = quantile(ah,[0.025,0.975])
        @test q[1] < sim.true_hard < q[2]
        @test abs(mean(ah) - sim.true_hard) < 0.12
        as = ate(fit, Soft(1,0.2)); qs = quantile(as,[0.025,0.975])
        @test qs[1] < sim.true_soft(1,0.2) < qs[2]
    end
end
