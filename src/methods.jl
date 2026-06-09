struct HCMFit
    β0::Matrix{Float64}
    β1::Matrix{Float64}
    p::Matrix{Float64}
    spec::HCMSpec
    chain
end

ate(fit::HCMFit, intv; baseline=default_baseline(intv)) =
    ate(fit.β0, fit.β1, fit.p, fit.spec.family, intv; baseline=baseline)

function summarize_ate(draws::AbstractVector)
    q = quantile(draws, [0.025, 0.975])
    (; mean = mean(draws), std = std(draws), q025 = q[1], q975 = q[2])
end

function Base.show(io::IO, fit::HCMFit)
    s = summarize_ate(ate(fit, Hard(1); baseline=Hard(0)))
    print(io, "HCMFit(motif=$(fit.spec.motif), units=$(fit.spec.n_units), ",
              "hard ATE=$(round(s.mean,digits=3)) [$(round(s.q025,digits=3)), $(round(s.q975,digits=3))])")
end
