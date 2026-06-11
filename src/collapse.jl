# General identification engine for hierarchical causal models.
#
# Pipeline (Weinstein & Blei 2024, §4–5):
#
#   HCMGraph ──collapse()──▶ CollapsedModel ──augment()/marginalize()──▶
#       ──latent_project()──▶ CausalGraphs.ADMG ──identify()──▶ verdict + functional
#
# This automates the *graph* layer of the framework: collapsing the plated graph to a
# flat model over distribution-valued Q variables, projecting out the latents, and
# running do-calculus via CausalGraphs.jl. It does NOT supply the two non-graph layers
# that real applications turn on — domain mechanism constraints that make a latent
# observable (mark it observed here), and estimation of the identified functional.

import CausalGraphs: ADMG, make_graph, identify, to_mermaid

# ── HCMGraph: a plate-aware causal DAG with hidden nodes ──────────────────────
struct HCMGraph
    vertices::Vector{Symbol}
    level::Dict{Symbol,Symbol}            # :unit or :subunit
    observed::Dict{Symbol,Bool}
    directed_edges::Vector{Tuple{Symbol,Symbol}}
end

"""
    hcm_graph(; vertices, subunit=[], hidden=[], di_edges=[])

Build a hierarchical causal model graph. `subunit` lists the subunit-level (inner-plate)
vertices (the rest are unit-level); `hidden` lists the unobserved vertices (the rest are
observed); `di_edges` are directed `(from, to)` pairs.
"""
function hcm_graph(; vertices, subunit=Symbol[], hidden=Symbol[],
                   di_edges=Tuple{Symbol,Symbol}[])
    V   = collect(Symbol.(vertices))
    sub = Set(Symbol.(subunit)); hid = Set(Symbol.(hidden))
    level    = Dict(v => (v in sub ? :subunit : :unit) for v in V)
    observed = Dict(v => !(v in hid) for v in V)
    HCMGraph(V, level, observed, [(Symbol(a), Symbol(b)) for (a, b) in di_edges])
end

# ── CollapsedModel: a flat DAG (directed only) with explicit hidden nodes ──────
struct CollapsedModel
    vertices::Vector{Symbol}
    observed::Dict{Symbol,Bool}
    directed_edges::Vector{Tuple{Symbol,Symbol}}
    labels::Dict{Symbol,String}
end

"""
    collapsed_model(; vertices, hidden=[], di_edges=[], labels=Dict())

Build a collapsed (flat) model directly. Use this to encode a domain mechanism
constraint that makes a latent observable — e.g. CAIRE's selection law renders the
fitness `r` reconstructable, so `r` is listed as a *non*-hidden vertex here.
"""
function collapsed_model(; vertices, hidden=Symbol[], di_edges=Tuple{Symbol,Symbol}[],
                         labels=Dict{Symbol,String}())
    V   = collect(Symbol.(vertices)); hid = Set(Symbol.(hidden))
    obs = Dict(v => !(v in hid) for v in V)
    lab = Dict{Symbol,String}(Symbol(k) => String(val) for (k, val) in labels)
    for v in V; haskey(lab, v) || (lab[v] = String(v)); end
    CollapsedModel(V, obs, [(Symbol(a), Symbol(b)) for (a, b) in di_edges], lab)
end

# ── small graph helpers (local; ADMG has its own in CausalGraphs) ─────────────
_par(edges, v) = unique(Symbol[a for (a, b) in edges if b == v])
_chi(edges, v) = unique(Symbol[b for (a, b) in edges if a == v])

# ── collapse: paper Algorithm 1 ───────────────────────────────────────────────
_qsym(v) = Symbol("Q", v)

# subunit vars connected to unit `w` via directed paths through subunit-only intermediates
function _direct_subunit_ancestors(g::HCMGraph, w)
    dsa = Symbol[]; seen = Set{Symbol}()
    frontier = [p for p in _par(g.directed_edges, w) if g.level[p] == :subunit]
    while !isempty(frontier)
        nxt = Symbol[]
        for s in frontier
            s in seen && continue
            push!(seen, s); push!(dsa, s)
            for p in _par(g.directed_edges, s)
                g.level[p] == :subunit && !(p in seen) && push!(nxt, p)
            end
        end
        frontier = nxt
    end
    unique(dsa)
end

_lower(xs) = join(lowercase.(string.(xs)), ",")

"""
    collapse(g::HCMGraph) -> CollapsedModel

Collapse the plated graph (m → ∞): every subunit variable `v` becomes a unit-level
Q variable `Q^{v|paS(v)}` (its within-unit conditional given subunit parents), observed
iff `v` and its subunit parents are observed. Unit parents of `v` point into `Q_v`; each
unit variable keeps its unit parents and gains an edge from the `Q` of each direct subunit
ancestor. The inner plate is erased.
"""
function collapse(g::HCMGraph)
    subunits = [v for v in g.vertices if g.level[v] == :subunit]
    units    = [v for v in g.vertices if g.level[v] == :unit]
    V = Symbol[]; obs = Dict{Symbol,Bool}(); edges = Tuple{Symbol,Symbol}[]
    labels = Dict{Symbol,String}()
    for v in subunits
        q = _qsym(v); push!(V, q)
        subpar = [p for p in _par(g.directed_edges, v) if g.level[p] == :subunit]
        obs[q] = g.observed[v] && all(p -> g.observed[p], subpar)
        labels[q] = isempty(subpar) ? "Q^{$(lowercase(string(v)))}" :
                                      "Q^{$(lowercase(string(v)))|$(_lower(subpar))}"
        for p in _par(g.directed_edges, v)
            g.level[p] == :unit && push!(edges, (p, q))
        end
    end
    for w in units
        push!(V, w); obs[w] = g.observed[w]; labels[w] = String(w)
        for p in _par(g.directed_edges, w)
            g.level[p] == :unit && push!(edges, (p, w))
        end
        for v in _direct_subunit_ancestors(g, w)
            push!(edges, (_qsym(v), w))
        end
    end
    CollapsedModel(V, obs, unique(edges), labels)
end

# ── augment (Algorithm 2) and marginalize (Algorithm 3) ───────────────────────
"""
    augment(m; node, parents, redirect_to=nothing, label=nothing)

Add a derived (graph-deterministic) variable `node` with the given `parents`. If
`redirect_to` is set, edges from `parents` to that child are rerouted through `node`
(used to introduce `q^a` between the conditionals and the outcome in the instrument).
"""
function augment(m::CollapsedModel; node, parents, redirect_to=nothing, label=nothing)
    node = Symbol(node); pars = Symbol.(parents)
    V = copy(m.vertices); push!(V, node)
    obs = copy(m.observed); obs[node] = true
    edges = copy(m.directed_edges)
    for p in pars; push!(edges, (p, node)); end
    labels = copy(m.labels); labels[node] = label === nothing ? String(node) : String(label)
    if redirect_to !== nothing
        rt = Symbol(redirect_to)
        edges = [e for e in edges if !(e[2] == rt && e[1] in pars)]
        push!(edges, (node, rt))
    end
    CollapsedModel(V, obs, unique(edges), labels)
end

"""
    marginalize(m, node) -> CollapsedModel

Remove `node`, reconnecting each of its parents to each of its children. Used to make a
deterministic augmentation variable depend *stochastically* on its remaining parents,
restoring the positivity do-calculus needs (the instrument motif).
"""
function marginalize(m::CollapsedModel, node)
    node = Symbol(node)
    pars = _par(m.directed_edges, node); kids = _chi(m.directed_edges, node)
    edges = [e for e in m.directed_edges if e[1] != node && e[2] != node]
    for p in pars, k in kids; push!(edges, (p, k)); end
    V = [v for v in m.vertices if v != node]
    obs = Dict(v => m.observed[v] for v in V)
    labels = Dict(v => m.labels[v] for v in V)
    CollapsedModel(V, obs, unique(edges), labels)
end

# ── latent projection: collapsed model (with hidden nodes) -> ADMG ────────────
# observed vertices reachable from `start` via directed edges through hidden-only intermediates
function _boundary_reach(m::CollapsedModel, start)
    reach = Symbol[]; seen = Set{Symbol}()
    frontier = _chi(m.directed_edges, start)
    while !isempty(frontier)
        nxt = Symbol[]
        for c in frontier
            c in seen && continue; push!(seen, c)
            if m.observed[c]
                push!(reach, c)                       # boundary: stop here
            else
                append!(nxt, _chi(m.directed_edges, c))
            end
        end
        frontier = nxt
    end
    unique(reach)
end

"""
    latent_project(m::CollapsedModel) -> CausalGraphs.ADMG

Project out the hidden vertices. A directed edge `x→y` is kept when `y` is reachable from
an observed `x` through hidden-only intermediates; a bidirected edge `x↔y` is added for
every hidden vertex whose hidden-only directed reach contains both `x` and `y` (a common
latent cause). This is the standard latent projection — the one piece CausalGraphs.jl,
which represents hidden confounding only as bidirected edges, does not provide.
"""
function latent_project(m::CollapsedModel)
    obsv = [v for v in m.vertices if m.observed[v]]
    di = Tuple{Symbol,Symbol}[]; bi = Tuple{Symbol,Symbol}[]
    for x in obsv, y in _boundary_reach(m, x)
        push!(di, (x, y))
    end
    for h in m.vertices
        m.observed[h] && continue
        r = _boundary_reach(m, h)
        for i in 1:length(r), j in (i + 1):length(r)
            push!(bi, (r[i], r[j]))
        end
    end
    make_graph(vertices=obsv, di_edges=unique(di), bi_edges=unique(bi))
end

# ── motif recognition (so the user knows which existing estimator applies) ─────
function _perms(xs::Vector{Symbol})
    isempty(xs) && return [Symbol[]]
    out = Vector{Vector{Symbol}}()
    for i in eachindex(xs)
        rest = vcat(xs[1:i-1], xs[i+1:end])
        for p in _perms(rest); push!(out, vcat(xs[i], p)); end
    end
    out
end

# isomorphism with treatment→treatment and outcome→outcome pinned (graphs are tiny)
function _admg_iso(g1, t1, o1, g2, t2, o2)
    length(g1.vertices) == length(g2.vertices) || return false
    length(g1.directed_edges) == length(g2.directed_edges) || return false
    length(g1.bidirected_edges) == length(g2.bidirected_edges) || return false
    rest1 = [v for v in g1.vertices if v != t1 && v != o1]
    rest2 = [v for v in g2.vertices if v != t2 && v != o2]
    length(rest1) == length(rest2) || return false
    de2 = Set(g2.directed_edges)
    be2 = Set(Set{Symbol}([a, b]) for (a, b) in g2.bidirected_edges)
    for perm in _perms(rest2)
        map = Dict(t1 => t2, o1 => o2)
        for (a, b) in zip(rest1, perm); map[a] = b; end
        ok = all(((a, b),) -> (map[a], map[b]) in de2, g1.directed_edges) &&
             all(((a, b),) -> Set{Symbol}([map[a], map[b]]) in be2, g1.bidirected_edges)
        ok && return true
    end
    false
end

# reference ADMGs for the three implemented motifs, built through this very pipeline.
# Memoized: recognize_motif runs on every identify_hcm call, and the references are
# themselves built from motif graphs — recomputing (and recursing) must be avoided.
const _REF_MOTIFS = Ref{Any}(nothing)

function _reference_motifs()
    _REF_MOTIFS[] === nothing || return _REF_MOTIFS[]
    refs = [(:confounder, identify_admg(hcm_confounder())...),
            (:confounder_interference, identify_admg(hcm_interference())...),
            (:instrument, identify_admg(hcm_instrument())...)]
    _REF_MOTIFS[] = refs
end

"Return (admg, treatment, outcome) for a built-in motif graph via the pipeline (no motif recognition)."
function identify_admg(spec::NamedTuple)
    admg, _, _, _ = _pipeline(spec.graph; treatment=spec.treatment, outcome=spec.outcome,
                              augments=get(spec, :augments, []),
                              marginalizes=get(spec, :marginalizes, Symbol[]))
    (admg, Symbol(spec.treatment), Symbol(spec.outcome))
end

function recognize_motif(admg::ADMG, treatment::Symbol, outcome::Symbol)
    for (name, g2, t2, o2) in _reference_motifs()
        _admg_iso(admg, treatment, outcome, g2, t2, o2) && return name
    end
    :unknown
end

# ── built-in motif graphs ─────────────────────────────────────────────────────
"CONFOUNDER motif (Fig 2a): unit confounder U, subunit A, subunit Y. Augment q^y."
hcm_confounder() = (
    graph = hcm_graph(vertices=[:U, :A, :Y], subunit=[:A, :Y], hidden=[:U],
                      di_edges=[(:U, :A), (:U, :Y), (:A, :Y)]),
    treatment = :QA, outcome = :qy,
    augments = [(node=:qy, parents=[:QA, :QY], label="q^{y}")],
)

"CONFOUNDER & INTERFERENCE motif (Fig 2e): A → unit channel Z → all Y. Augment q^y."
hcm_interference() = (
    graph = hcm_graph(vertices=[:U, :A, :Z, :Y], subunit=[:A, :Y], hidden=[:U],
                      di_edges=[(:U, :A), (:U, :Y), (:A, :Y), (:A, :Z), (:Z, :Y)]),
    treatment = :QA, outcome = :qy,
    augments = [(node=:qy, parents=[:QA, :QY], label="q^{y}")],
)

"INSTRUMENT motif (Fig 2i): subunit instrument Z → A, unit outcome Y. Augment q^a, marginalize q^z."
hcm_instrument() = (
    graph = hcm_graph(vertices=[:U, :Z, :A, :Y], subunit=[:Z, :A], hidden=[:U],
                      di_edges=[(:Z, :A), (:U, :A), (:U, :Y), (:A, :Y)]),
    treatment = :Qa, outcome = :Y,
    augments = [(node=:Qa, parents=[:QZ, :QA], redirect_to=:Y, label="q^{a}")],
    marginalizes = [:QZ],
)

"CAIRE collapsed repertoire-IV model (Weinstein et al. 2024). `r_observed=true` encodes the
selection-law mechanism constraint that makes fitness reconstructable; `false` drops it."
function caire_model(; r_observed::Bool=true)
    hidden = r_observed ? [:U] : [:U, :r]
    collapsed_model(
        vertices = [:U, :r, :qz, :qa, :y], hidden = hidden,
        di_edges = [(:U, :r), (:U, :y), (:r, :qa), (:qz, :qa), (:qa, :y)],
        labels = Dict(:qz => "q^{z}", :qa => "q^{a}", :r => "r", :y => "y"),
    )
end

# ── main entry point ──────────────────────────────────────────────────────────
"""
    identify_hcm(x; treatment, outcome, augments=[], marginalizes=[])

Run the identification pipeline on an `HCMGraph` (collapsed automatically) or a
`CollapsedModel` (used as-is). `augments` is a vector of NamedTuples splatted into
[`augment`](@ref); `marginalizes` is a vector of node symbols passed to [`marginalize`](@ref),
applied in order. Returns a NamedTuple:

  - `verdict`        — `:a_fixable`, `:p_fixable`, `:nested_fixable`, `:id_algorithm`, or `:not_identified`
  - `identification` — the full `CausalGraphs.identify` result (functional, adjustment sets, …)
  - `admg`           — the projected `CausalGraphs.ADMG`
  - `model`          — the final `CollapsedModel`
  - `stages`         — `[(name, model_or_admg), …]` for each pipeline stage (for diagrams)
  - `motif`          — `:confounder`, `:confounder_interference`, `:instrument`, or `:unknown`
"""
# Core pipeline: collapse → augment → marginalize → project → identify. No motif lookup.
function _pipeline(x; treatment, outcome, augments=[], marginalizes=Symbol[])
    m = x isa HCMGraph ? collapse(x) : x::CollapsedModel
    stages = Any[("collapsed", m)]
    for a in augments
        m = augment(m; a...)
    end
    isempty(augments) || push!(stages, ("augmented", m))
    for nd in marginalizes
        m = marginalize(m, nd)
    end
    isempty(marginalizes) || push!(stages, ("marginalized", m))
    admg = latent_project(m)
    push!(stages, ("admg", admg))
    res = identify(admg, Symbol(treatment), Symbol(outcome))
    (admg, m, stages, res)
end

function identify_hcm(x; treatment, outcome, augments=[], marginalizes=Symbol[])
    admg, m, stages, res = _pipeline(x; treatment=treatment, outcome=outcome,
                                     augments=augments, marginalizes=marginalizes)
    motif = recognize_motif(admg, Symbol(treatment), Symbol(outcome))
    (verdict=res.strategy, identification=res, admg=admg, model=m, stages=stages, motif=motif)
end

# ── mermaid rendering ─────────────────────────────────────────────────────────
_san(v) = replace(string(v), r"[^A-Za-z0-9_]" => "_")

function to_mermaid(g::HCMGraph; direction="LR")
    io = IOBuffer(); println(io, "flowchart $direction")
    subs = [v for v in g.vertices if g.level[v] == :subunit]
    units = [v for v in g.vertices if g.level[v] == :unit]
    println(io, "  subgraph unit_plate[\"unit i\"]")
    for v in units
        shape = g.observed[v] ? ("[\"", "\"]") : ("([\"", " (hidden)\"])")
        println(io, "    $(_san(v))$(shape[1])$(v)$(shape[2])")
    end
    println(io, "    subgraph subunit_plate[\"subunit j\"]")
    for v in subs
        shape = g.observed[v] ? ("[\"", "\"]") : ("([\"", " (hidden)\"])")
        println(io, "      $(_san(v))$(shape[1])$(v)$(shape[2])")
    end
    println(io, "    end")
    println(io, "  end")
    for (a, b) in g.directed_edges
        println(io, "  $(_san(a)) --> $(_san(b))")
    end
    String(take!(io))
end

function to_mermaid(m::CollapsedModel; direction="LR")
    io = IOBuffer(); println(io, "flowchart $direction")
    for v in m.vertices
        lab = m.labels[v]
        shape = m.observed[v] ? ("[\"", "\"]") : ("([\"", " (hidden)\"])")
        println(io, "  $(_san(v))$(shape[1])$(lab)$(shape[2])")
    end
    for (a, b) in m.directed_edges
        println(io, "  $(_san(a)) --> $(_san(b))")
    end
    String(take!(io))
end
