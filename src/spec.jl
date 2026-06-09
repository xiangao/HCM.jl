struct HCMSpec
    outcome::Symbol
    treatment::Symbol
    unit_id::Vector{Int}
    n_units::Int
    y::Vector{Float64}
    a::Vector{Int}
    family::Symbol
    motif::Symbol
end

function hcm_spec(formula::FormulaTerm, data; unit::Symbol, subunit::Symbol,
                  motif::Symbol=:confounder, family::Symbol=:gaussian)
    outcome   = StatsModels.termvars(formula.lhs)[1]
    treatment = StatsModels.termvars(formula.rhs)[1]
    a = Int.(data[!, treatment])
    all(x -> x in (0,1), a) || error("treatment must be 0/1 in A1")
    codes = sort(unique(data[!, unit]))
    idx = Dict(c => i for (i,c) in enumerate(codes))
    unit_id = [idx[c] for c in data[!, unit]]
    HCMSpec(outcome, treatment, unit_id, length(codes),
            Float64.(data[!, outcome]), a, family, motif)
end
