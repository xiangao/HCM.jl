@model function nested_model(y, a, school, class_, n_school, n_class, abar_class_i, abar_school_i, family)
    β0 ~ Normal(0,5); β1 ~ Normal(0,5); λclass ~ Normal(0,5); λschool ~ Normal(0,5)
    σschool ~ truncated(Normal(0,2); lower=0); σclass ~ truncated(Normal(0,2); lower=0)
    zs ~ filldist(Normal(0,1), n_school); zc ~ filldist(Normal(0,1), n_class)
    πc ~ filldist(Normal(0,3), n_class)
    b_school = σschool .* zs; b_class = σclass .* zc
    N = length(y)
    for i in 1:N
        a[i] ~ Bernoulli(logistic(πc[class_[i]]))
    end
    if family === :gaussian
        σy ~ truncated(Normal(0,2); lower=0)
        for i in 1:N
            η = β0 + β1*a[i] + λclass*abar_class_i[i] + λschool*abar_school_i[i] + b_school[school[i]] + b_class[class_[i]]
            y[i] ~ Normal(η, σy)
        end
    else
        for i in 1:N
            η = β0 + β1*a[i] + λclass*abar_class_i[i] + λschool*abar_school_i[i] + b_school[school[i]] + b_class[class_[i]]
            y[i] ~ BernoulliLogit(η)
        end
    end
    return (; β0, β1, λclass, λschool, σschool, σclass, b_school, b_class, p_class = logistic.(πc))
end

@model function interference_model(y, a, unit, z, abar, s, n_units, family)
    μ0 ~ Normal(0,5); μ1 ~ Normal(0,5); δ0 ~ Normal(0,5); δ1 ~ Normal(0,5)
    γ0 ~ Normal(0,5); γ1 ~ Normal(0,5); γ2 ~ Normal(0,5)
    σz ~ truncated(Normal(0,2); lower=0); τ0 ~ truncated(Normal(0,2); lower=0)
    πlogit ~ filldist(Normal(0,3), n_units); zu ~ filldist(Normal(0,1), n_units); u = τ0 .* zu
    N = length(y)
    for i in 1:N
        a[i] ~ Bernoulli(logistic(πlogit[unit[i]]))
    end
    for j in 1:n_units
        z[j] ~ Normal(γ0 + γ1*abar[j] + γ2*s[j], σz)
    end
    if family === :gaussian
        σy ~ truncated(Normal(0,2); lower=0)
        for i in 1:N
            η = μ0 + u[unit[i]] + δ0*z[unit[i]] + (μ1 + δ1*z[unit[i]])*a[i]
            y[i] ~ Normal(η, σy)
        end
    else
        for i in 1:N
            η = μ0 + u[unit[i]] + δ0*z[unit[i]] + (μ1 + δ1*z[unit[i]])*a[i]
            y[i] ~ BernoulliLogit(η)
        end
    end
    return (; μ0, μ1, δ0, δ1, γ0, γ1, γ2, σz, u, p = logistic.(πlogit))
end

@model function instrument_model(y, a, z, unit, n_units, family)
    μω ~ Normal(0,2); σω ~ truncated(Normal(0,2); lower=0)
    zω ~ filldist(Normal(0,1), n_units); ω = logistic.(μω .+ σω .* zω)
    μα ~ Normal(0,2); σα ~ truncated(Normal(0,2); lower=0)
    μβ ~ Normal(0,2); σβ ~ truncated(Normal(0,2); lower=0)
    zα ~ filldist(Normal(0,1), n_units); zβ ~ filldist(Normal(0,1), n_units)
    α = μα .+ σα .* zα; β = μβ .+ σβ .* zβ
    θ0 ~ Normal(0,5); θa ~ Normal(0,5); θr0 ~ Normal(0,5); θr1 ~ Normal(0,5)
    N = length(a)
    for i in 1:N
        z[i] ~ Bernoulli(ω[unit[i]])
        a[i] ~ Bernoulli(logistic(α[unit[i]] + β[unit[i]]*z[i]))
    end
    π0 = logistic.(α); π1 = logistic.(α .+ β)
    qa = ω .* π1 .+ (1 .- ω) .* π0
    if family === :gaussian
        σy ~ truncated(Normal(0,2); lower=0)
        for k in 1:n_units
            y[k] ~ Normal(θ0 + θa*qa[k] + θr0*π0[k] + θr1*π1[k], σy)
        end
    else
        for k in 1:n_units
            y[k] ~ BernoulliLogit(θ0 + θa*qa[k] + θr0*π0[k] + θr1*π1[k])
        end
    end
    return (; θ0, θa, θr0, θr1, π0, π1, qa)
end

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
