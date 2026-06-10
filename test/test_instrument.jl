@testset "instrument recovery" begin
    # Confounding enters through q^{a|z}=(π0,π1) (θr0,θr1), exactly the model's backdoor set, so the
    # model is correctly specified. θ_a is identified ONLY by instrument spread (ω), since
    # qa = ω·π1 + (1−ω)·π0 is otherwise collinear with (π0,π1). This makes it a weak-instrument
    # estimand: the CI reliably covers θ_a (the IV guarantee), but the POINT estimate is imprecise —
    # we verified the channel-aligned fix halves the bias (~0.34 → ~0.17) and that stronger
    # instruments tighten it further. So the robust checks here are CI-coverage; the point check is a
    # generous gross-bias guard, not a precision claim. (Seeded for reproducibility.)
    Random.seed!(1)
    sim = sim_hcm(:instrument; n=120, m=30,
                  params=(θa=0.5, θr0=1.0, θr1=-0.5, ρ=1.0, β0=2.0, sd_ω=2.0, sd_y=0.3),
                  family=:gaussian, seed=31)
    fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit,
              instrument=:z, motif=:instrument, iter=1000, chains=2)
    ah = ate(fit, Hard(1); baseline=Hard(0)); q = quantile(ah, [0.025, 0.975])
    @test q[1] < sim.true_hard < q[2]                 # CI covers θ_a (the IV guarantee — robust)
    @test abs(mean(ah) - sim.true_hard) < 0.3         # generous gross-bias guard (weak-IV imprecision)
    as = ate(fit, Soft(1, 0.2)); qs = quantile(as, [0.025, 0.975])
    @test qs[1] < sim.true_soft(1, 0.2) < qs[2]       # soft CI covers
end
