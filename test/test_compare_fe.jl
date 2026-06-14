@testset "compare_fe" begin
    if get(ENV, "HCM_SAMPLING_TESTS", "") != "1"
        @test_skip "set HCM_SAMPLING_TESTS=1 to run NUTS recovery tests"
    else
        sim = sim_hcm(:confounder; n=40, m=30, params=(β1=0.5, ρ=1.0), family=:gaussian, seed=7)
        fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit, iter=400, chains=2)
        tab = compare_fe(fit)
        @test Set(tab.method) == Set(["HCM (posterior mean)", "Unit FE"])
        fe = tab.estimate[tab.method .== "Unit FE"][1]
        @test abs(fe - sim.true_hard) < 0.1
    end
end
