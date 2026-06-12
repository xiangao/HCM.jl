"""
    compare_fe(fit::HCMFit) -> NamedTuple

Compare the HCM estimate against a classical fixed-effects / OLS baseline for the same data:
for `:confounder`/`:nested_confounder`/`:confounder_interference` the within-unit (two-way)
FE slope, and for `:instrument` the naive OLS of the unit outcome on the observed treatment
rate `q^a` (no `q^{a|z}` control). Useful for seeing where HCM and the SUTVA/no-spillover
baseline agree or diverge.
"""
function compare_fe(fit::HCMFit)
    sp = fit.spec
    if sp.motif === :did
        # FE-DiD baseline: student + school×year fixed effects (= the HCM layer in the linear case).
        # For a binary outcome this is the linear-probability model, biased for the ATE.
        df = DataFrame(y=sp.y, D=Float64.(sp.D), student=sp.student_id, sy=sp.sy_id)
        m = reg(df, @formula(y ~ D + fe(student) + fe(sy)))
        fe_est = coef(m)[findfirst(==("D"), coefnames(m))]
        return DataFrame(method=["HCM did (posterior mean)", "FE-DiD (student + school×year)"],
                         estimate=[mean(ate(fit, Hard(1); baseline=Hard(0))), fe_est])
    end
    if sp.motif === :instrument
        # unit-level outcome: naive OLS of y on the observed treatment rate q^a (no q^{a|z} control);
        # biased by U because qa correlates with the confounded (π0,π1). Unit FE doesn't apply (1 y/unit).
        udf = DataFrame(y = sp.y, qa = sp.qa_obs)
        m = reg(udf, @formula(y ~ qa))
        naive = coef(m)[findfirst(==("qa"), coefnames(m))]
        return DataFrame(method = ["HCM (backdoor θ_a)", "Naive OLS (y~q^a)"],
                         estimate = [mean(ate(fit, Hard(1); baseline=Hard(0))), naive])
    end
    if sp.motif === :nested_confounder
        df = DataFrame(y=sp.y, a=Float64.(sp.a), school=sp.school_id, class=sp.class_id)
        m = reg(df, @formula(y ~ a + fe(school) + fe(class)))
        fe_est = coef(m)[findfirst(==("a"), coefnames(m))]
        return DataFrame(method=["HCM (posterior mean)","Two-way FE"],
                         estimate=[mean(ate(fit, Hard(1); baseline=Hard(0))), fe_est])
    end
    if sp.motif === :confounder_interference
        df = DataFrame(y=sp.y, a=Float64.(sp.a), unit=sp.unit_id)
        m = reg(df, @formula(y ~ a + fe(unit)))
        fe_est = coef(m)[findfirst(==("a"), coefnames(m))]
        return DataFrame(method=["HCM (posterior mean)","Unit FE"],
                         estimate=[mean(ate(fit, Hard(1); baseline=Hard(0))), fe_est])
    end
    df = DataFrame(y = sp.y, a = Float64.(sp.a), unit = sp.unit_id)
    m = reg(df, @formula(y ~ a + fe(unit)))
    fe_est = coef(m)[findfirst(==("a"), coefnames(m))]
    hcm_est = mean(ate(fit, Hard(1); baseline=Hard(0)))
    DataFrame(method = ["HCM (posterior mean)", "Unit FE"],
              estimate = [hcm_est, fe_est])
end
