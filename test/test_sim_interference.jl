@testset "sim interference" begin
    out = sim_hcm(:confounder_interference; n=40, m=20,
                  params=(μ0=0.0, μ1=0.3, δ0=0.2, δ1=0.6, γ0=0.0, γ1=-1.0, γ2=0.2,
                          σz=0.4, τ0=1.0, sd_y=0.3, sd_s=1.0, p_a=0.5), family=:gaussian, seed=7)
    @test Set(names(out.data)) == Set(["unit","subunit","a","y","z","s"])
    @test length(out.u) == 40 && length(out.p) == 40
    @test isfinite(out.true_hard) && isfinite(out.true_soft(1,0.2))
end
