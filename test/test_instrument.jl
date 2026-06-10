@testset "instrument recovery" begin
    sim = sim_hcm(:instrument; n=120, m=30,
                  params=(θa=0.5, ψ=0.5, ρ=0.5, β0=1.5, sd_ω=1.5, sd_y=0.3),
                  family=:gaussian, seed=31)
    fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit,
              instrument=:z, motif=:instrument, iter=1000, chains=2)
    ah = ate(fit, Hard(1); baseline=Hard(0)); q = quantile(ah,[0.025,0.975])
    @test q[1] < sim.true_hard < q[2]
    @test abs(mean(ah) - sim.true_hard) < 0.15
    as = ate(fit, Soft(1,0.2)); qs = quantile(as,[0.025,0.975])
    @test qs[1] < sim.true_soft(1,0.2) < qs[2]
end
