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

struct InterferenceSpec
    outcome::Symbol; treatment::Symbol
    y::Vector{Float64}; a::Vector{Int}
    unit_id::Vector{Int}; n_units::Int
    abar::Vector{Float64}; abar_i::Vector{Float64}
    s::Vector{Float64}; s_i::Vector{Float64}
    z::Vector{Float64}
    family::Symbol; motif::Symbol
end

struct InstrumentSpec
    outcome::Symbol; treatment::Symbol; instrument::Symbol
    y::Vector{Float64}              # length n_units (unit-level)
    a::Vector{Int}; z::Vector{Int}  # length N (subunit)
    unit_id::Vector{Int}; n_units::Int
    qa_obs::Vector{Float64}         # length n_units
    family::Symbol; motif::Symbol
end

function hcm_spec(formula::FormulaTerm, data; unit::Symbol=:unit, subunit::Symbol=:subunit,
                  motif::Symbol=:confounder, family::Symbol=:gaussian, groups=nothing,
                  interferer=nothing, unit_covar=nothing, instrument=nothing)
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
    if motif === :confounder_interference
        outcome = StatsModels.termvars(formula.lhs)[1]; treatment = StatsModels.termvars(formula.rhs)[1]
        a = Int.(data[!, treatment]); all(x->x in (0,1), a) || error("treatment must be 0/1")
        y = Float64.(data[!, outcome])
        ucodes = sort(unique(data[!, unit])); unit_id = Int.(indexin(data[!, unit], ucodes))
        nU = length(ucodes)
        abar = [mean(a[unit_id .== i]) for i in 1:nU]
        # interferer z and covariate s are unit-level: take first value per unit, validate constant
        unitfirst(col) = [first(data[!, col][unit_id .== i]) for i in 1:nU]
        isunitlevel(col) = all(i -> length(unique(data[!, col][unit_id .== i])) == 1, 1:nU)
        interferer === nothing && error("motif :confounder_interference requires `interferer`")
        isunitlevel(interferer) || error("interferer must be unit-level (constant within unit)")
        z = Float64.(unitfirst(interferer))
        if unit_covar === nothing
            s = zeros(nU)
        else
            isunitlevel(unit_covar) || error("unit_covar must be unit-level")
            s = Float64.(unitfirst(unit_covar))
        end
        return InterferenceSpec(outcome, treatment, y, a, unit_id, nU,
                                abar, abar[unit_id], s, s[unit_id], z, family, motif)
    end
    if motif === :instrument
        outcome = StatsModels.termvars(formula.lhs)[1]; treatment = StatsModels.termvars(formula.rhs)[1]
        instrument === nothing && error("motif :instrument requires `instrument`")
        a = Int.(data[!, treatment]); all(x->x in (0,1), a) || error("treatment must be 0/1")
        z = Int.(data[!, instrument]); all(x->x in (0,1), z) || error("instrument must be 0/1")
        ucodes = sort(unique(data[!, unit])); unit_id = Int.(indexin(data[!, unit], ucodes)); nU = length(ucodes)
        ycol = data[!, outcome]
        all(i -> length(unique(ycol[unit_id .== i]))==1, 1:nU) || error("outcome must be unit-level (constant within unit)")
        y = Float64[first(ycol[unit_id .== i]) for i in 1:nU]
        qa_obs = [mean(a[unit_id .== i]) for i in 1:nU]
        return InstrumentSpec(outcome, treatment, instrument, y, a, z, unit_id, nU, qa_obs, family, motif)
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
