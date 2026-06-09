@testset "nested estimand" begin
    spec = (motif=:nested_confounder, family=:gaussian,
            school_id=[1,1,2,2], class_id=[1,1,2,2],
            abar_class_i=[0.5,0.5,0.5,0.5], abar_school_i=[0.5,0.5,0.5,0.5], p_class_idx=[1,1,2,2])
    draws = (β0=[0.0,0.0], β1=[0.4,0.8], λclass=[0.0,0.0], λschool=[0.0,0.0],
             b_school=[0.0 0.0; 0.0 0.0], b_class=[0.0 0.0; 0.0 0.0], p_class=[0.3 0.7; 0.3 0.7])
    @test HCM._ate(Val(:nested_confounder), draws, spec, Hard(1); baseline=Hard(0)) ≈ [0.4, 0.8]
    as = HCM._ate(Val(:nested_confounder), draws, spec, Soft(1, 0.2))
    exp1 = 0.2*mean(0.4 .* (1 .- [0.3,0.3,0.7,0.7]))
    exp2 = 0.2*mean(0.8 .* (1 .- [0.3,0.3,0.7,0.7]))
    @test as ≈ [exp1, exp2]
end
