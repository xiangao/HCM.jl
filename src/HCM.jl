module HCM
using Distributions, DataFrames, StatsModels, StatsFuns, Statistics, Random, LinearAlgebra
using Turing, MCMCChains, FixedEffectModels

include("intervention.jl")
export Hard, Soft

include("estimand.jl")
export ate, link_inv

include("sim.jl")
export sim_hcm

include("spec.jl")
export hcm_spec, HCMSpec

include("models.jl")

include("methods.jl")
export hcm, HCMFit, summarize_ate

function hcm(formula, data; unit::Symbol, subunit::Symbol,
             motif::Symbol=:confounder, family::Symbol=:gaussian,
             chains::Int=2, iter::Int=600, kwargs...)
    spec  = hcm_spec(formula, data; unit=unit, subunit=subunit, motif=motif, family=family)
    model = confounder_model(spec.y, spec.a, spec.unit_id, spec.n_units, family)
    # Julia has 1 thread on this machine; use MCMCSerial for multi-chain
    chain = sample(model, NUTS(0.95), MCMCSerial(), iter, chains; progress=false, kwargs...)
    gq    = generated_quantities(model, chain)
    D     = length(gq)
    J     = spec.n_units
    β0   = Matrix{Float64}(undef, D, J)
    β1   = similar(β0)
    pmat  = similar(β0)
    for (d, q) in enumerate(vec(gq))
        β0[d, :]   = q.β[1, :]
        β1[d, :]   = q.β[2, :]
        pmat[d, :] = q.p
    end
    HCMFit(β0, β1, pmat, spec, chain)
end

end # module
