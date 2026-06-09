@testset "spec interference" begin
    d = sim_hcm(:confounder_interference; n=8, m=10, seed=3).data
    sp = hcm_spec(@formula(y ~ a), d; unit=:unit, subunit=:subunit,
                  motif=:confounder_interference, interferer=:z, unit_covar=:s)
    @test sp.motif == :confounder_interference
    @test sp.n_units == 8
    @test length(sp.abar_i) == nrow(d) && length(sp.s_i) == nrow(d)
    @test length(sp.abar) == 8 && length(sp.s) == 8 && length(sp.z) == 8
end
