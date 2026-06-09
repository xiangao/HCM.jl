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

struct NestedSpec
    outcome::Symbol; treatment::Symbol
    y::Vector{Float64}; a::Vector{Int}
    school_id::Vector{Int}; class_id::Vector{Int}
    n_schools::Int; n_classes::Int
    abar_class::Vector{Float64}; abar_school::Vector{Float64}
    abar_class_i::Vector{Float64}; abar_school_i::Vector{Float64}
    p_class_idx::Vector{Int}
    family::Symbol; motif::Symbol
end

function hcm_spec(formula::FormulaTerm, data; unit::Symbol=:unit, subunit::Symbol=:subunit,
                  motif::Symbol=:confounder, family::Symbol=:gaussian, groups=nothing)
    if motif === :nested_confounder
        outcome = StatsModels.termvars(formula.lhs)[1]; treatment = StatsModels.termvars(formula.rhs)[1]
        a = Int.(data[!, treatment]); all(x->x in (0,1), a) || error("treatment must be 0/1")
        y = Float64.(data[!, outcome])
        sch, cls = groups
        school_id = Int.(indexin(data[!,sch], sort(unique(data[!,sch]))))
        classkey = string.(data[!,sch], "|", data[!,cls])
        codes = sort(unique(classkey)); class_id = Int.(indexin(classkey, codes))
        K = maximum(school_id); C = maximum(class_id)
        abar_class = [mean(a[class_id .== c]) for c in 1:C]
        abar_school = [mean(a[school_id .== k]) for k in 1:K]
        return NestedSpec(outcome, treatment, y, a, school_id, class_id, K, C,
                          abar_class, abar_school, abar_class[class_id], abar_school[school_id],
                          class_id, family, motif)
    end
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
