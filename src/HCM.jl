module HCM
using Distributions, DataFrames, StatsModels, StatsFuns, Statistics, Random, LinearAlgebra
using Turing, MCMCChains, FixedEffectModels
using FastGaussQuadrature

include("intervention.jl")
export Hard, Soft

include("estimand.jl")
export ate, link_inv

include("sim.jl")
export sim_hcm

include("spec.jl")
export hcm_spec, HCMSpec, NestedSpec

include("models.jl")

include("methods.jl")
export hcm, HCMFit, summarize_ate, nested_diagnostics

include("compare_fe.jl")
export compare_fe

function hcm(formula, data; unit::Symbol=:unit, subunit::Symbol=:subunit,
             motif::Symbol=:confounder, family::Symbol=:gaussian,
             chains::Int=2, iter::Int=600, groups=nothing, kwargs...)
    spec  = hcm_spec(formula, data; unit=unit, subunit=subunit, motif=motif, family=family, groups=groups)
    if motif === :nested_confounder
        model = nested_model(spec.y, spec.a, spec.school_id, spec.class_id, spec.n_schools, spec.n_classes,
                             spec.abar_class_i, spec.abar_school_i, family)
        chain = sample(model, NUTS(0.9), MCMCSerial(), iter, chains; progress=false, kwargs...)
        gq = vec(generated_quantities(model, chain))
        draws = (β0=[q.β0 for q in gq], β1=[q.β1 for q in gq],
                 λclass=[q.λclass for q in gq], λschool=[q.λschool for q in gq],
                 σschool=[q.σschool for q in gq], σclass=[q.σclass for q in gq],
                 b_school=reduce(vcat, [permutedims(q.b_school) for q in gq]),
                 b_class=reduce(vcat, [permutedims(q.b_class) for q in gq]),
                 p_class=reduce(vcat, [permutedims(q.p_class) for q in gq]))
        return HCMFit(draws, spec, chain)
    end
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
    HCMFit((; β0=β0, β1=β1, p=pmat), spec, chain)
end

end # module
