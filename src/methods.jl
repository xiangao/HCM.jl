struct HCMFit
    β0::Matrix{Float64}
    β1::Matrix{Float64}
    p::Matrix{Float64}
    spec::HCMSpec
    chain
end

ate(fit::HCMFit, intv; baseline=default_baseline(intv)) =
    ate(fit.β0, fit.β1, fit.p, fit.spec.family, intv; baseline=baseline)
