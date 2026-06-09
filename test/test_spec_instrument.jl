@testset "spec instrument" begin
    d = sim_hcm(:instrument; n=8, m=20, seed=2).data
    sp = hcm_spec(@formula(y ~ a), d; unit=:unit, subunit=:subunit, instrument=:z, motif=:instrument)
    @test sp.motif == :instrument
    @test sp.n_units == 8
    @test length(sp.y) == 8
    @test length(sp.a) == nrow(d) && length(sp.z) == nrow(d)
    @test length(sp.qa_obs) == 8
    @test sp.qa_obs ≈ [mean(sp.a[sp.unit_id .== i]) for i in 1:8]
end
