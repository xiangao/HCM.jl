@testset "confounder recovery" begin
    sim = sim_hcm(:confounder; n=50, m=40,
                  params=(β1=0.5, sd_u=1.0, ρ=1.0, sd_y=0.3), family=:gaussian, seed=11)
    fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit,
              motif=:confounder, family=:gaussian, iter=600, chains=2)
    ah = ate(fit, Hard(1); baseline=Hard(0))
    q  = quantile(ah, [0.025, 0.975])
    @test q[1] < sim.true_hard < q[2]
    @test abs(mean(ah) - sim.true_hard) < 0.1
    as  = ate(fit, Soft(1, 0.2))
    qs  = quantile(as, [0.025, 0.975])
    @test qs[1] < sim.true_soft(1, 0.2) < qs[2]
end
