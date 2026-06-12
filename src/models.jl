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

# Joint HCM + DiD: student random effect (DiD/partial-pooling layer) + school×year random effect
# (HCM within-group layer, absorbs all confounding constant within school-year, time-varying included)
# + treatment effect τ. Identified because the treatment varies within school-year (below the
# saturated δ). gaussian or bernoulli (the nonlinear margin). Returns per-draw components so the
# estimand can form the ATT on the correct scale.
@model function did_model(y, D, Dbar, student, sy, n_students, n_sy, family)
    τ  ~ Normal(0, 5)
    λ  ~ Normal(0, 5)                              # Mundlak / correlated-random-effects correction
    σα ~ truncated(Normal(0, 2); lower = 0)        # student RE sd — partial pooling (the value-add)
    zα ~ filldist(Normal(0, 1), n_students); α = σα .* zα
    # school×year effects are the confounder-absorbing layer: keep them UNPOOLED (fixed-effect-like,
    # wide prior) so a strong upper-level confounder is fully removed, as fixed effects would.
    δ ~ filldist(Normal(0, 10), n_sy)
    # λ·Dbar (each student's mean treatment) de-biases the *pooled* student RE: without it this is a
    # random-effects DiD, biased when adoption is correlated with α (Mundlak 1978 / Hausman). With it,
    # τ equals the within (fixed-effects) estimate while retaining partial pooling on α.
    η = α[student] .+ δ[sy] .+ τ .* D .+ λ .* Dbar
    if family === :gaussian
        σy ~ truncated(Normal(0, 2); lower = 0)
        for i in eachindex(y)
            y[i] ~ Normal(η[i], σy)
        end
    else
        for i in eachindex(y)
            y[i] ~ BernoulliLogit(η[i])
        end
    end
    return (; τ, λ, σα, α, δ)
end

# Interference over time: peer channel z_{gt} driven by the within-school treatment RATE feeds back
# onto every student's outcome. School confounding is STRUCTURED (b_g + trend_g·t), NOT saturated
# school×year — saturating would absorb the channel. Student selection handled by the RE + Mundlak λ.
# Identifies the DIRECT effect (a student's own treatment, channel held) vs the TOTAL effect (treat
# everyone → rate→1 → channel shifts → all outcomes) — which FE-DiD conflates. Gaussian.
@model function did_interf_model(y, D, Dbar, z, abar, s, student, sy, school_of_sy, period_of_sy,
                                 n_students, n_schools, n_sy)
    τ  ~ Normal(0, 5); λ ~ Normal(0, 5)                 # direct effect + Mundlak correction
    δ0 ~ Normal(0, 5); δ1 ~ Normal(0, 5)                # channel → outcome (peer)
    γ0 ~ Normal(0, 5); γ1 ~ Normal(0, 5); γ2 ~ Normal(0, 5)
    σz ~ truncated(Normal(0, 2); lower = 0)
    σα ~ truncated(Normal(0, 2); lower = 0)
    zα ~ filldist(Normal(0, 1), n_students); α = σα .* zα
    b  ~ filldist(Normal(0, 5), n_schools)              # school level (fixed-like)
    tr ~ filldist(Normal(0, 2), n_schools)              # school trend (structured confounder)
    for j in 1:n_sy                                     # channel model, per school×year cell
        z[j] ~ Normal(γ0 + γ1 * abar[j] + γ2 * s[j], σz)
    end
    σy ~ truncated(Normal(0, 2); lower = 0)
    for i in eachindex(y)
        c = sy[i]
        η = α[student[i]] + b[school_of_sy[c]] + tr[school_of_sy[c]] * period_of_sy[c] +
            δ0 * z[c] + (τ + δ1 * z[c]) * D[i] + λ * Dbar[i]
        y[i] ~ Normal(η, σy)
    end
    return (; τ, λ, δ0, δ1, γ0, γ1, γ2, σz, α, b, tr)
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
