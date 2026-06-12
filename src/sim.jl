"""
    sim_hcm(motif; n, m, params=NamedTuple(), family=:gaussian, seed=nothing)

Simulate data from a hierarchical causal model with `n` units and `m` subunits each, for
`motif` ∈ `(:confounder, :nested_confounder, :confounder_interference, :instrument)`.
Override DGP parameters via `params` (e.g. `params=(γ1=-1.2,)` to set the interference
strength). Returns a NamedTuple with `data` (long-format `DataFrame`) and the analytic
ground-truth effects `true_hard` and `true_soft(a★, ε)` for that DGP, plus the latent
variables — so estimates can be checked against truth.
"""
function sim_hcm(motif::Symbol; n, m, params=NamedTuple(), family::Symbol=:gaussian, seed=nothing)
    motif === :confounder || motif === :nested_confounder || motif === :confounder_interference ||
        motif === :instrument || motif === :did || motif === :did_interference ||
        error("sim_hcm supports :confounder, :nested_confounder, :confounder_interference, :instrument, :did, :did_interference")
    seed === nothing || Random.seed!(seed)
    if motif === :did_interference
        # interference over time: within-school treatment RATE drives a school×year channel z that
        # feeds back onto every outcome. School confounding is STRUCTURED (b_g + trend_g·t).
        p_ = merge((T=6, τ=0.3, δ0=0.4, δ1=0.3, γ0=0.0, γ1=-0.8, γ2=0.2, σz=0.3,
                    σα=1.0, sd_b=0.7, sd_tr=0.3, sd_scov=1.0, sd_y=0.3, sel_α=0.5, sel_tr=2.0), params)
        G = n; mps = m; T = p_.T; Ns = G * mps
        α = randn(Ns) .* p_.σα; school_of = repeat(1:G, inner=mps)
        b = randn(G) .* p_.sd_b; trend = randn(G) .* p_.sd_tr; scov = randn(G) .* p_.sd_scov
        lin = p_.sel_α .* α .+ p_.sel_tr .* trend[school_of] .+ randn(Ns)
        qn = (lin .- minimum(lin)) ./ (maximum(lin) - minimum(lin) + 1e-9)
        adopt = 2 .+ floor.(Int, (1 .- qn) .* (T - 1)); adopt[qn .< 0.25] .= T + 1
        student = repeat(1:Ns, inner=T); period = repeat(1:T, outer=Ns); school = school_of[student]
        D = Int.(period .>= adopt[student])
        # within-school-year rate and channel
        sykey = string.(school, "|", period); sycodes = sort(unique(sykey)); sy = Int.(indexin(sykey, sycodes))
        nsy = length(sycodes)
        abar = [mean(D[sy .== c]) for c in 1:nsy]
        sch_of_sy = [school[findfirst(==(c), sy)] for c in 1:nsy]
        z = p_.γ0 .+ p_.γ1 .* abar .+ p_.γ2 .* scov[sch_of_sy] .+ randn(nsy) .* p_.σz
        η = α[student] .+ b[school] .+ trend[school] .* period .+ p_.δ0 .* z[sy] .+
            (p_.τ .+ p_.δ1 .* z[sy]) .* D
        y = η .+ randn(length(student)) .* p_.sd_y
        data = DataFrame(student=student, school=school, period=period, a=D, y=y,
                         z=z[sy], scov=scov[school])
        # true effects (averaged over the realized sample)
        true_direct = p_.τ + p_.δ1 * mean(z[sy])                       # within-D, channel held
        z1 = p_.γ0 .+ p_.γ1 .* 1.0 .+ p_.γ2 .* scov[sch_of_sy]         # channel if everyone treated
        z0 = p_.γ0 .+ p_.γ1 .* 0.0 .+ p_.γ2 .* scov[sch_of_sy]         # channel if no one treated
        true_total = p_.δ0 * (mean(z1) - mean(z0)) + p_.τ + p_.δ1 * mean(z1)   # full-do total
        return (; data, true_total, true_hard=true_total, true_direct, α, b, trend, scov)
    end
    if motif === :did
        # Panel: students in schools over T periods. Time-varying school confounder = school level b_g
        # + school TREND s_g (the part plain school FE misses). Staggered student adoption correlated
        # with α and s (confounded). gaussian or bernoulli (the nonlinear margin).
        p_ = merge((T=6, β0=0.0, τ=0.5, σα=1.0, sd_b=0.7, sd_s=0.3, sd_y=0.5,
                    sel_α=0.6, sel_s=2.0), params)
        G = n; mps = m; T = p_.T; Ns = G * mps
        α = randn(Ns) .* p_.σα
        school_of = repeat(1:G, inner=mps)
        b = randn(G) .* p_.sd_b; s = randn(G) .* p_.sd_s
        lin = p_.sel_α .* α .+ p_.sel_s .* s[school_of] .+ randn(Ns)
        qn = (lin .- minimum(lin)) ./ (maximum(lin) - minimum(lin) + 1e-9)
        adopt = 2 .+ floor.(Int, (1 .- qn) .* (T - 1))      # in 2..T
        adopt[qn .< 0.25] .= T + 1                          # 25% never-treated controls
        student = repeat(1:Ns, inner=T); period = repeat(1:T, outer=Ns)
        school = school_of[student]
        δ = b[school] .+ s[school] .* (period .- (T + 1) / 2)
        D = Int.(period .>= adopt[student])
        η0 = p_.β0 .+ α[student] .+ δ
        if family === :gaussian
            y = η0 .+ p_.τ .* D .+ randn(length(student)) .* p_.sd_y
            true_hard = float(p_.τ)
        else
            y = Float64.(rand(length(student)) .< logistic.(η0 .+ p_.τ .* D))
            true_hard = mean(logistic.(η0 .+ p_.τ) .- logistic.(η0))   # average marginal effect (prob scale)
        end
        data = DataFrame(student=student, school=school, period=period, a=D, y=y)
        return (; data, true_hard, α, b, s, adopt)
    end
    if motif === :instrument
        # Confounding of Y enters THROUGH q^{a|z}=(π0,π1) (the backdoor set the model adjusts for),
        # matching the HCM instrument identification — U → (α,π0,π1) → both treatment and outcome.
        p_ = merge((θ0=0.0, θa=0.5, θr0=1.0, θr1=-0.5, sd_u=1.0,
                    α0=0.0, ρ=1.0, sd_α=0.3, β0=1.0, sd_β=0.3, sd_y=0.3, sd_ω=1.0), params)
        u = randn(n).*p_.sd_u
        ω = logistic.(randn(n).*p_.sd_ω)
        α = p_.α0 .+ p_.ρ.*u .+ randn(n).*p_.sd_α
        β = p_.β0 .+ randn(n).*p_.sd_β
        π0 = logistic.(α); π1 = logistic.(α .+ β)
        unit = repeat(1:n, inner=m); subunit = repeat(1:m, outer=n)
        z = Int.(rand(n*m) .< ω[unit])
        a = Int.(rand(n*m) .< logistic.(α[unit] .+ β[unit].*z))
        qa = ω.*π1 .+ (1 .- ω).*π0
        B = p_.θ0 .+ p_.θr0.*π0 .+ p_.θr1.*π1            # per-unit baseline (confounding via q^{a|z})
        ηy = B .+ p_.θa.*qa
        y_unit = family === :gaussian ? ηy .+ randn(n).*p_.sd_y : Float64.(rand(n) .< logistic.(ηy))
        data = DataFrame(unit=unit, subunit=subunit, z=z, a=a, y=y_unit[unit])
        gi = x -> logistic(x)
        if family === :gaussian
            true_hard = p_.θa
            true_soft = (a★, ε) -> p_.θa * ε * mean(a★ .- qa)
        else
            true_hard = mean(gi.(B .+ p_.θa) .- gi.(B))
            true_soft = (a★, ε) -> begin
                rate_new = (1-ε).*qa .+ ε.*a★
                mean(gi.(B .+ p_.θa.*rate_new) .- gi.(B .+ p_.θa.*qa))
            end
        end
        return (; data, true_hard, true_soft, u, ω, π0, π1, qa)
    end
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
        edo(1, 0, fill(1.0, n)) - edo(0, 0, fill(0.0, n))   # full do: ā=1 vs ā=0 through the channel
    else
        abar_new = (1-ε).*abar .+ ε.*a★
        edo(a★, ε, abar_new) - edo(a★, 0, abar)
    end
end
