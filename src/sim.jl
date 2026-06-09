function sim_hcm(motif::Symbol; n, m, params=NamedTuple(), family::Symbol=:gaussian, seed=nothing)
    motif === :confounder || motif === :nested_confounder || error("A1 implements only :confounder")
    seed === nothing || Random.seed!(seed)
    if motif === :nested_confounder
        p_ = merge((classes_per_school=4, β0=0.0, β1=0.5, sd_school=1.0, sd_class=0.7,
                    sd_y=0.3, ρ_school=1.0, ρ_class=1.0), params)
        K = n; L = p_.classes_per_school; C = K * L
        b_school = randn(K) .* p_.sd_school
        b_class  = randn(C) .* p_.sd_class
        class_school = repeat(1:K, inner=L)
        class_id = repeat(1:C, inner=m)
        school_id = class_school[class_id]
        student = repeat(1:m, outer=C)
        p_class = logistic.(p_.ρ_school .* b_school[class_school] .+ p_.ρ_class .* b_class)  # length C
        a = Int.(rand(C*m) .< p_class[class_id])
        η = p_.β0 .+ b_school[school_id] .+ b_class[class_id] .+ p_.β1 .* a
        y = family === :gaussian ? η .+ randn(C*m).*p_.sd_y : Float64.(rand(C*m) .< logistic.(η))
        data = DataFrame(school=school_id, class=class_id, student=student, a=a, y=y)
        if family === :gaussian
            true_hard = p_.β1
            true_soft = (a★, ε) -> ε * mean(p_.β1 .* (a★ .- p_class))
        else
            gi = x -> logistic(x)
            B = p_.β0 .+ b_school[class_school] .+ b_class    # per-class baseline (length C)
            true_hard = mean(gi.(B .+ p_.β1) .- gi.(B))
            true_soft = (a★, ε) -> ε*mean(gi.(B .+ p_.β1 .* a★) .- (p_class.*gi.(B .+ p_.β1) .+ (1 .- p_class).*gi.(B)))
        end
        return (; data, true_hard, true_soft, b_school, b_class, p_class)
    end
    p_ = merge((β0=0.0, β1=0.5, sd_u=1.0, α_p=0.0, ρ=1.0, sd_y=0.3), params)
    u = randn(n) .* p_.sd_u
    prop = logistic.(p_.α_p .+ p_.ρ .* u)
    unit = repeat(1:n, inner=m)
    subunit = repeat(1:m, outer=n)
    a = Int.(rand(n*m) .< prop[unit])
    η = p_.β0 .+ u[unit] .+ p_.β1 .* a
    y = family === :gaussian ? η .+ randn(n*m) .* p_.sd_y :
        Float64.(rand(n*m) .< logistic.(η))
    data = DataFrame(unit=unit, subunit=subunit, a=a, y=y)
    if family === :gaussian
        true_hard = p_.β1
        true_soft = (a_star, ε) -> ε * mean(p_.β1 .* (a_star .- prop))
    else
        gi = x -> logistic(x)
        true_hard = mean(gi.(p_.β0 .+ u .+ p_.β1) .- gi.(p_.β0 .+ u))
        true_soft = (a_star, ε) -> begin
            treated = gi.(p_.β0 .+ u .+ p_.β1); control = gi.(p_.β0 .+ u)
            star = gi.(p_.β0 .+ u .+ p_.β1 .* a_star)
            obs = prop .* treated .+ (1 .- prop) .* control
            mean(((1-ε) .* obs .+ ε .* star) .- obs)
        end
    end
    (; data, true_hard, true_soft, u, p=prop)
end
