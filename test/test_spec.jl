@testset "spec" begin
    d = sim_hcm(:confounder; n=10, m=20, seed=2).data
    sp = hcm_spec(@formula(y ~ a), d; unit=:unit, subunit=:subunit, motif=:confounder, family=:gaussian)
    @test sp.outcome == :y && sp.treatment == :a
    @test sp.n_units == 10
    @test length(sp.unit_id) == nrow(d)
    @test sort(unique(sp.unit_id)) == collect(1:10)
    @test all(x -> x in (0,1), sp.a)
end
