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
