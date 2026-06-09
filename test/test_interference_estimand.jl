@testset "interference estimand" begin
    spec = (motif=:confounder_interference, family=:bernoulli,
            n_units=2, abar=[0.5,0.5], abar_i=[0.5,0.5], s_i=[0.0,0.0], s=[0.0,0.0])
    draws = (μ0=[0.1], μ1=[0.4], δ0=[0.2], δ1=[0.5], γ0=[0.2], γ1=[-0.9], γ2=[0.0],
             σz=[0.5], u=[0.3 -0.4], p=[0.5 0.5])   # 1 draw × 2 units
    as = HCM._ate(Val(:confounder_interference), draws, spec, Soft(1, 0.1))
    Random.seed!(1); Z=400000; gi(x)=1/(1+exp(-x))
    edo = function(a★, ε, abar_eff)
        mean(map(1:2) do i
            zmean = 0.2 - 0.9*abar_eff[i] + 0.0*0.0
            z = zmean .+ 0.5 .* randn(Z)
            my(av) = gi.(0.1 .+ draws.u[1,i] .+ 0.2 .* z .+ (0.4 .+ 0.5 .* z) .* av)
            mean((1-ε).*(0.5 .* my(1) .+ 0.5 .* my(0)) .+ ε .* my(a★))
        end)
    end
    abar_new = (1-0.1).*[0.5,0.5] .+ 0.1.*1
    @test as[1] ≈ (edo(1,0.1,abar_new) - edo(1,0,[0.5,0.5])) atol=2e-3
end
