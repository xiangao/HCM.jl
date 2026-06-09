@testset "instrument estimand" begin
    spec = (motif=:instrument, family=:gaussian, qa_obs=[0.3,0.6])
    draws = (θ0=[0.0,0.0], θa=[0.5,1.0], θr0=[0.0,0.0], θr1=[0.0,0.0],
             π0=[0.0 0.0; 0.0 0.0], π1=[0.0 0.0; 0.0 0.0])
    @test HCM._ate(Val(:instrument), draws, spec, Hard(1); baseline=Hard(0)) ≈ [0.5, 1.0]
    as = HCM._ate(Val(:instrument), draws, spec, Soft(1, 0.2))
    f = mean(1 .- [0.3,0.6])
    @test as ≈ [0.5*0.2*f, 1.0*0.2*f]
    # bernoulli hard vs hand reference (1 draw × 2 units)
    specb = (motif=:instrument, family=:bernoulli, qa_obs=[0.5,0.5])
    drb = (θ0=[0.1], θa=[0.6], θr0=[0.2], θr1=[-0.1], π0=[0.4 0.7], π1=[0.5 0.8])
    ahb = HCM._ate(Val(:instrument), drb, specb, Hard(1); baseline=Hard(0))
    B = [0.1 + 0.2*0.4 + (-0.1)*0.5, 0.1 + 0.2*0.7 + (-0.1)*0.8]
    @test ahb ≈ [mean(@. 1/(1+exp(-(B+0.6))) - 1/(1+exp(-B)))]
end
