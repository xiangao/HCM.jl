struct HCMFit
    draws::NamedTuple
    spec        # untyped: HCMSpec now; NestedSpec/InterferenceSpec later. dispatch via spec.motif
    chain
end

ate(fit::HCMFit, intv; baseline=default_baseline(intv)) =
    _ate(Val(fit.spec.motif), fit.draws, fit.spec, intv; baseline=baseline)

function summarize_ate(draws::AbstractVector)
    q = quantile(draws, [0.025, 0.975])
    (; mean = mean(draws), std = std(draws), q025 = q[1], q975 = q[2])
end

function Base.show(io::IO, fit::HCMFit)
    s = summarize_ate(ate(fit, Hard(1); baseline=Hard(0)))
    print(io, "HCMFit(motif=$(fit.spec.motif), units=$(fit.spec.n_units), ",
              "hard ATE=$(round(s.mean,digits=3)) [$(round(s.q025,digits=3)), $(round(s.q975,digits=3))])")
end
