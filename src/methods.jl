struct HCMFit
    draws::NamedTuple
    spec        # untyped: HCMSpec now; NestedSpec/InterferenceSpec later. dispatch via spec.motif
    chain
end

"""
    ate(fit::HCMFit, intv; baseline=default_baseline(intv)) -> Vector

Posterior draws of the average treatment effect of intervention `intv` versus `baseline`,
computed by pushing each posterior draw through the motif's identification formula
(dispatched on `fit.spec.motif`). `intv` is a [`Hard`](@ref) or [`Soft`](@ref) intervention;
the default baseline is the corresponding "do nothing" arm (`Hard(0)` / `Soft(a★, 0)`).
Summarize with [`summarize_ate`](@ref).
"""
ate(fit::HCMFit, intv; baseline=default_baseline(intv)) =
    _ate(Val(fit.spec.motif), fit.draws, fit.spec, intv; baseline=baseline)

"""
    summarize_ate(draws) -> NamedTuple

Posterior `(mean, std, q025, q975)` of a vector of ATE draws from [`ate`](@ref).
"""
function summarize_ate(draws::AbstractVector)
    q = quantile(draws, [0.025, 0.975])
    (; mean = mean(draws), std = std(draws), q025 = q[1], q975 = q[2])
end

"""
    nested_diagnostics(draws) -> NamedTuple

For a `:nested_confounder` fit, posterior summaries of the Mundlak group-mean coefficients
`λclass`, `λschool` (a Hausman-style confounding diagnostic — far from zero ⇒ between-group
variation is confounded) and the variance components `σschool`, `σclass`.
"""
function nested_diagnostics(draws)
    summ(v) = (; mean=mean(v), q025=quantile(v,0.025), q975=quantile(v,0.975))
    (; λclass=summ(draws.λclass), λschool=summ(draws.λschool),
       σschool=summ(draws.σschool), σclass=summ(draws.σclass))
end

function Base.show(io::IO, fit::HCMFit)
    s = summarize_ate(ate(fit, Hard(1); baseline=Hard(0)))
    print(io, "HCMFit(motif=$(fit.spec.motif), units=$(fit.spec.n_units), ",
              "hard ATE=$(round(s.mean,digits=3)) [$(round(s.q025,digits=3)), $(round(s.q975,digits=3))])")
end
