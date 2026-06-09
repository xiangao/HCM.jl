@model function confounder_model(y, a, unit, n_units, family)
    μ  ~ MvNormal(zeros(2), 25.0 * I(2))
    τ  ~ filldist(truncated(Normal(0, 2); lower=0), 2)
    μp ~ Normal(0, 2)
    σp ~ truncated(Normal(0, 2); lower=0)
    zβ ~ filldist(Normal(0, 1), 2, n_units)
    zp ~ filldist(Normal(0, 1), n_units)
    β  = μ .+ τ .* zβ
    logitp = μp .+ σp .* zp

    # DynamicPPL ≥ v0.35: arrays of distributions must use loops, not .~
    for i in eachindex(a)
        a[i] ~ Bernoulli(logistic(logitp[unit[i]]))
    end

    η = β[1, unit] .+ β[2, unit] .* a

    if family === :gaussian
        σy ~ truncated(Normal(0, 2); lower=0)
        for i in eachindex(y)
            y[i] ~ Normal(η[i], σy)
        end
    else
        for i in eachindex(y)
            y[i] ~ BernoulliLogit(η[i])
        end
    end

    return (; β, p = logistic.(logitp))
end
