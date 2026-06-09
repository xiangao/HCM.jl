link_inv(η, family::Symbol) = family === :gaussian ? η :
                              family === :bernoulli ? logistic.(η) :
                              error("unknown family: $family")

# per-draw E[Y; do(intv)], averaged over units. β0,β1,p are draws × n_units.
function expected_outcome(β0, β1, p, family, intv::Hard)
    vec(mean(link_inv(β0 .+ β1 .* intv.a_star, family); dims=2))
end

function expected_outcome(β0, β1, p, family, intv::Soft)
    treated = link_inv(β0 .+ β1, family)
    control = link_inv(β0, family)
    star    = link_inv(β0 .+ β1 .* intv.a_star, family)
    obs_mix = p .* treated .+ (1 .- p) .* control
    do_mix  = (1 - intv.ε) .* obs_mix .+ intv.ε .* star
    vec(mean(do_mix; dims=2))
end

function ate(β0, β1, p, family, intv; baseline=default_baseline(intv))
    expected_outcome(β0, β1, p, family, intv) .- expected_outcome(β0, β1, p, family, baseline)
end

function _ate(::Val{:confounder}, draws, spec, intv; baseline=default_baseline(intv))
    ate(draws.β0, draws.β1, draws.p, spec.family, intv; baseline=baseline)
end

function _ate(::Val{:nested_confounder}, draws, spec, intv; baseline=default_baseline(intv))
    _eo_nested(draws, spec, intv) .- _eo_nested(draws, spec, baseline)
end

function _ate(::Val{:confounder_interference}, draws, spec, intv; baseline=default_baseline(intv))
    _eo_interf(draws, spec, intv) .- _eo_interf(draws, spec, baseline)
end

function _eo_interf(draws, spec, intv; gh=gausshermite(24))
    nodes, w = gh; D = length(draws.μ0); fam = spec.family
    abar = spec.abar; s = spec.s; n = spec.n_units
    out = zeros(D)
    for d in 1:D
        # Hard do(A=a★): set the whole unit to a★, so ā=a★ propagates through the
        #   z-channel (the full interference effect — the motif's point over FE).
        # Soft do(σ_{a★,ε}): dose ε at a★, shifting the unit share ā proportionally.
        abar_eff = intv isa Soft ? ((1-intv.ε).*abar .+ intv.ε.*intv.a_star) :
                                   fill(float(intv.a_star), n)
        acc_units = 0.0
        for i in 1:n
            zmean = draws.γ0[d] + draws.γ1[d]*abar_eff[i] + draws.γ2[d]*s[i]
            ai = 0.0
            for k in eachindex(nodes)
                z = zmean + sqrt(2)*draws.σz[d]*nodes[k]
                my = av -> link_inv(draws.μ0[d] + draws.u[d,i] + draws.δ0[d]*z + (draws.μ1[d] + draws.δ1[d]*z)*av, fam)
                val = intv isa Hard ? my(intv.a_star) :
                      (1-intv.ε)*(draws.p[d,i]*my(1) + (1-draws.p[d,i])*my(0)) + intv.ε*my(intv.a_star)
                ai += (w[k]/sqrt(pi))*val
            end
            acc_units += ai
        end
        out[d] = acc_units / n
    end
    out
end

function _ate(::Val{:instrument}, draws, spec, intv; baseline=default_baseline(intv))
    _eo_instrument(draws, spec, intv) .- _eo_instrument(draws, spec, baseline)
end

function _eo_instrument(draws, spec, intv)
    D = length(draws.θa); fam = spec.family; qaobs = spec.qa_obs
    out = zeros(D)
    for d in 1:D
        B = draws.θ0[d] .+ draws.θr0[d].*draws.π0[d,:] .+ draws.θr1[d].*draws.π1[d,:]
        rate = intv isa Hard ? fill(float(intv.a_star), length(B)) :
                               (1-intv.ε).*qaobs .+ intv.ε.*intv.a_star
        out[d] = mean(link_inv(B .+ draws.θa[d].*rate, fam))
    end
    out
end

function _eo_nested(draws, spec, intv)
    D = length(draws.β1); fam = spec.family
    ki = spec.school_id; ci = spec.class_id
    out = zeros(D)
    for d in 1:D
        B = draws.β0[d] .+ draws.λclass[d].*spec.abar_class_i .+ draws.λschool[d].*spec.abar_school_i .+
            draws.b_school[d, ki] .+ draws.b_class[d, ci]
        pc = draws.p_class[d, spec.p_class_idx]
        if intv isa Hard
            out[d] = mean(link_inv(B .+ draws.β1[d].*intv.a_star, fam))
        else
            treated = link_inv(B .+ draws.β1[d], fam); control = link_inv(B, fam)
            star = link_inv(B .+ draws.β1[d].*intv.a_star, fam)
            obs = pc.*treated .+ (1 .- pc).*control
            out[d] = mean((1-intv.ε).*obs .+ intv.ε.*star)
        end
    end
    out
end
