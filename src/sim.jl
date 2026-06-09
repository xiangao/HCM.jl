function sim_hcm(motif::Symbol; n, m, params=NamedTuple(), family::Symbol=:gaussian, seed=nothing)
    motif === :confounder || error("A1 implements only :confounder")
    seed === nothing || Random.seed!(seed)
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
        gi(x) = logistic(x)
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
