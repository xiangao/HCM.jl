@testset "interference recovery" begin
    sim = sim_hcm(:confounder_interference; n=60, m=30,
                  params=(μ1=0.3, δ1=0.6, γ1=-1.0, σz=0.4, τ0=1.0, sd_y=0.3, p_a=0.5),
                  family=:gaussian, seed=21)
    fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit,
              motif=:confounder_interference, interferer=:z, unit_covar=:s, iter=600, chains=2)
    ah = ate(fit, Hard(1); baseline=Hard(0)); q = quantile(ah,[0.025,0.975])
    @test q[1] < sim.true_hard < q[2]
    as = ate(fit, Soft(1,0.2)); qs = quantile(as,[0.025,0.975])
    @test qs[1] < sim.true_soft(1,0.2) < qs[2]
end
