function compare_fe(fit::HCMFit)
    sp = fit.spec
    if sp.motif === :nested_confounder
        df = DataFrame(y=sp.y, a=Float64.(sp.a), school=sp.school_id, class=sp.class_id)
        m = reg(df, @formula(y ~ a + fe(school) + fe(class)))
        fe_est = coef(m)[findfirst(==("a"), coefnames(m))]
        return DataFrame(method=["HCM (posterior mean)","Two-way FE"],
                         estimate=[mean(ate(fit, Hard(1); baseline=Hard(0))), fe_est])
    end
    df = DataFrame(y = sp.y, a = Float64.(sp.a), unit = sp.unit_id)
    m = reg(df, @formula(y ~ a + fe(unit)))
    fe_est = coef(m)[findfirst(==("a"), coefnames(m))]
    hcm_est = mean(ate(fit, Hard(1); baseline=Hard(0)))
    DataFrame(method = ["HCM (posterior mean)", "Unit FE"],
              estimate = [hcm_est, fe_est])
end
