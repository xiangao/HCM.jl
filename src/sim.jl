function sim_hcm(motif::Symbol; n, m, params=NamedTuple(), family::Symbol=:gaussian, seed=nothing)
    motif === :confounder || motif === :nested_confounder || motif === :confounder_interference ||
        error("A1 implements only :confounder")
    seed === nothing || Random.seed!(seed)
    if motif === :confounder_interference
        p_ = merge((μ0=0.0, μ1=0.3, δ0=0.2, δ1=0.6, γ0=0.0, γ1=-0.8, γ2=0.2,
                    σz=0.5, τ0=1.0, sd_y=0.3, sd_s=1.0, p_a=0.5), params)
        u = randn(n).*p_.τ0; s = randn(n).*p_.sd_s; pri = fill(p_.p_a, n)
        unit = repeat(1:n, inner=m); subunit = repeat(1:m, outer=n)
        a = Int.(rand(n*m) .< pri[unit])
        abar = [mean(a[unit .== i]) for i in 1:n]
        z = p_.γ0 .+ p_.γ1 .* abar .+ p_.γ2 .* s .+ randn(n).*p_.σz
        η = p_.μ0 .+ u[unit] .+ p_.δ0 .* z[unit] .+ (p_.μ1 .+ p_.δ1 .* z[unit]) .* a
        y = family === :gaussian ? η .+ randn(n*m).*p_.sd_y : Float64.(rand(n*m) .< logistic.(η))
        data = DataFrame(unit=unit, subunit=subunit, a=a, y=y, z=z[unit], s=s[unit])
        true_hard = _interf_true(p_, u, s, pri, abar, family, :hard, 1, 0)
        true_soft = (a★, ε) -> _interf_true(p_, u, s, pri, abar, family, :soft, a★, ε)
        return (; data, true_hard, true_soft, u, p=pri, s)
    end
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

function _interf_true(p_, u, s, pri, abar, family, kind, a★, ε)
    nodes, w = gausshermite(24); n = length(u)
    gi = x -> family === :gaussian ? x : logistic(x)
    edo = function(astar, ε_, abar_eff)
        tot = 0.0
        for i in 1:n
            zmean = p_.γ0 + p_.γ1*abar_eff[i] + p_.γ2*s[i]
            acc = 0.0
            for k in eachindex(nodes)
                z = zmean + sqrt(2)*p_.σz*nodes[k]
                my = av -> gi(p_.μ0 + u[i] + p_.δ0*z + (p_.μ1 + p_.δ1*z)*av)
                val = kind === :hard ? my(astar) :
                      (1-ε_)*(pri[i]*my(1) + (1-pri[i])*my(0)) + ε_*my(astar)
                acc += (w[k]/sqrt(pi))*val
            end
            tot += acc
        end
        tot/n
    end
    if kind === :hard
        edo(1, 0, abar) - edo(0, 0, abar)
    else
        abar_new = (1-ε).*abar .+ ε.*a★
        edo(a★, ε, abar_new) - edo(a★, 0, abar)
    end
end
