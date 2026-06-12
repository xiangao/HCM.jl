using HCM, Test, DataFrames, StatsModels, Statistics, StatsFuns

@testset "did spec" begin
    d = DataFrame(student=repeat(1:4, inner=3), school=repeat([1,1,2,2], inner=3),
                  period=repeat(1:3, outer=4), a=[0,0,1, 0,1,1, 0,0,0, 0,0,1], y=randn(12))
    sp = hcm_spec(@formula(y ~ a), d; unit=:student, groups=(:school,:period), motif=:did)
    @test sp isa DiDSpec
    @test sp.n_students == 4
    @test sp.n_sy == 6                      # 2 schools × 3 periods
    @test all(x -> x in (0,1), sp.D)
    @test length(sp.y) == 12
end

@testset "did sim" begin
    g = sim_hcm(:did; n=10, m=8, seed=1, params=(T=5, τ=0.5))
    @test g.true_hard ≈ 0.5                 # gaussian additive ATT = τ
    @test Set(names(g.data)) ⊇ Set(["student","school","period","a","y"])
    @test all(x -> x in (0,1), g.data.a)
    b = sim_hcm(:did; n=10, m=8, seed=1, params=(T=5, τ=0.8), family=:bernoulli)
    @test 0.0 < b.true_hard < 1.0           # AME on the probability scale
end

@testset "did_interference sim + spec" begin
    s = sim_hcm(:did_interference; n=12, m=6, seed=1, params=(T=5,))
    @test s.true_total != s.true_direct            # spillover makes total ≠ direct
    @test Set(names(s.data)) ⊇ Set(["student","school","period","a","y","z","scov"])
    sp = hcm_spec(@formula(y ~ a), s.data; unit=:student, groups=(:school,:period),
                  motif=:did_interference, interferer=:z, unit_covar=:scov)
    @test sp isa DiDInterferenceSpec
    @test sp.n_sy == 12 * 5                          # 12 schools × 5 periods
    @test length(sp.z_cell) == sp.n_sy
    # estimand: total via Hard(1)/Hard(0); direct via did_direct — deterministic on a fake fit
    draws = (τ=[0.3,0.3], λ=[0.0,0.0], δ0=[0.4,0.4], δ1=[0.3,0.3],
             γ0=[0.0,0.0], γ1=[-0.8,-0.8], γ2=[0.2,0.2])
    fit = HCMFit(draws, sp, nothing)
    tot = ate(fit, Hard(1); baseline=Hard(0))
    sc = sp.scov_cell
    @test tot[1] ≈ 0.4*((mean(-0.8 .+ 0.2 .* sc)) - mean(0.2 .* sc)) + 0.3 + 0.3*mean(-0.8 .+ 0.2 .* sc)
    @test all(did_direct(fit) .≈ 0.3 + 0.3*mean(sp.z_cell))
end

@testset "did estimand (deterministic)" begin
    # tiny spec: 2 students, 2 sy cells, 4 obs; Dbar=0 so the Mundlak term drops out (η0=0)
    sp = DiDSpec(:y, :a, zeros(4), [0,1,0,1], zeros(4), [1,1,2,2], 2, [1,2,1,2], 2, [1,1,2,2], :gaussian, :did)
    D = 5
    draws = (τ=collect(0.1:0.1:0.5), λ=zeros(D), σα=ones(D), α=zeros(D,2), δ=zeros(D,2))
    eff = HCM._ate(Val(:did), draws, sp, Hard(1.0); baseline=Hard(0.0))
    @test eff ≈ draws.τ                                   # gaussian: additive τ
    spb = DiDSpec(:y, :a, zeros(4), [0,1,0,1], zeros(4), [1,1,2,2], 2, [1,2,1,2], 2, [1,1,2,2], :bernoulli, :did)
    ame = HCM._ate(Val(:did), draws, spb, Hard(1.0); baseline=Hard(0.0))
    @test ame ≈ logistic.(draws.τ) .- 0.5                 # bernoulli: AME at η0=0
end
