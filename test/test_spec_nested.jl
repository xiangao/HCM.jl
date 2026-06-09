@testset "spec nested" begin
    d = sim_hcm(:nested_confounder; n=6, m=5, params=(classes_per_school=3,), seed=2).data
    sp = hcm_spec(@formula(y ~ a), d; groups=(:school,:class), motif=:nested_confounder)
    @test sp.motif == :nested_confounder
    @test sp.n_schools == 6 && sp.n_classes == 18
    @test length(sp.school_id) == nrow(d) && length(sp.class_id) == nrow(d)
    @test length(sp.abar_class) == 18 && length(sp.abar_school) == 6
end
