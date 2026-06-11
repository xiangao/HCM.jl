# HCM.jl

[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://xiangao.github.io/HCM.jl/)

Hierarchical causal models in Julia (Weinstein & Blei 2024), estimated with Turing.
Four motifs — **confounder**, **confounder & interference**, **nested**, **instrument** — each with
**hard and soft (stochastic) interventions**, plus a **general identification engine** that collapses
an arbitrary HCM graph to an ADMG and runs do-calculus via
[CausalGraphs.jl](https://github.com/xiangao/CausalGraphs.jl).

**Documentation: <https://xiangao.github.io/HCM.jl/>** — vignettes for the motifs, the identification
engine, and an interference-vs-SUTVA showcase.

```julia
using HCM, DataFrames, StatsModels
sim = sim_hcm(:confounder; n=50, m=40, seed=1)
fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit, motif=:confounder)
ate(fit, Hard(1); baseline=Hard(0))   # hard ATE
ate(fit, Soft(1, 0.1))                # soft: add a*=1 at dosage 0.1
compare_fe(fit)                        # fixed-effects baseline
```

Soft interventions (`do(q^a ~ (1-ε)q^a + ε·δ_{a*})`) are the capability the R `hcm` package lacks;
the ATE is linear in the dosage `ε`. Roadmap: A2 (interference, nested, instrument motifs),
then a collapse-engine front-end to CausalGraphs.jl.

## More motifs (A2a)

```julia
# nested (students in classes in schools), Mundlak correlated random effects
fit = hcm(@formula(y ~ a), students; groups=(:school,:class), motif=:nested_confounder)
ate(fit, Soft(1, 0.1)); nested_diagnostics(fit.draws)   # within-class soft ATE + λ diagnostics

# interference (treatment acts through a unit-level channel z)
fit = hcm(@formula(y ~ a), audits; unit=:cell, subunit=:factory,
          motif=:confounder_interference, interferer=:z, unit_covar=:s)
ate(fit, Hard(1); baseline=Hard(0))   # full-do interference ATE (propagated through the channel)
ate(fit, Soft(1, 0.1))                # soft: dose a*=1 at ε through the front-door channel

# instrument (subunit IV z, unit-level outcome y; HCM do(q^a) rate effect, A2b)
fit = hcm(@formula(y ~ a), trials; unit=:site, subunit=:patient,
          instrument=:z, motif=:instrument)
ate(fit, Hard(1); baseline=Hard(0))   # effect of the unit treatment rate θ_a, backdoor-adjusted on q^{a|z}
compare_fe(fit)                        # HCM backdoor θ_a vs naive (biased) OLS of y on q^a
# NOTE: θ_a is a weak-instrument estimand — its CI reliably covers the truth, but the point
# estimate is imprecise unless the instrument is strong (large within-unit instrument spread).
```

## General identification engine

The motifs above are worked examples, not a fixed menu. `identify_hcm` applies the paper's
identification procedure (collapse → augment/marginalize → do-calculus) to an *arbitrary* HCM
graph, by collapsing it to a flat ADMG and routing to [`CausalGraphs.jl`](https://github.com/xiangao/CausalGraphs.jl):

```julia
using HCM
# build a plate-graph: which vertices are subunit-level, which are hidden, and the edges
g = hcm_graph(vertices=[:U,:A,:Y], subunit=[:A,:Y], hidden=[:U],
              di_edges=[(:U,:A),(:U,:Y),(:A,:Y)])             # the confounder motif
r = identify_hcm(g; treatment=:QA, outcome=:qy,
                 augments=[(node=:qy, parents=[:QA,:QY])])    # augment the outcome marginal q^y
r.verdict   # :a_fixable  (backdoor on Q^{y|a})
r.motif     # :confounder (which built-in estimator applies)
r.admg      # the projected CausalGraphs.ADMG
```

Verdicts are `:a_fixable` (backdoor), `:p_fixable` (front-door), `:nested_fixable`, `:id_algorithm`,
or `:not_identified`. The three motifs round-trip exactly (`hcm_confounder()`, `hcm_interference()`,
`hcm_instrument()` are built-in graphs).

### The engine is identification-only, by design

A real HCM application has three layers, and only the first is graph-determined:

1. **collapse + do-calculus → verdict + functional** — generalizable; this is what `identify_hcm` automates.
2. **mechanism constraints that make a latent observable** — *not* graph-determined; a domain modelling choice.
3. **estimation of the identified functional** — bespoke per data type.

The flagship application **CAIRE** (Weinstein, Wood & Blei 2024, T-cell receptors; instrument motif)
makes this concrete. Its identification turns on a population-genetics *selection law*
`qᵃ = r·qᶻ / Σ r·qᶻ`, which reconstructs the latent fitness `r` from the two observed repertoire
distributions — so `r` becomes effectively observed. In the engine that is a one-word change:

```julia
identify_hcm(caire_model(r_observed=true);  treatment=:qa, outcome=:y).verdict  # :a_fixable  (backdoor on r — paper Thm 1)
identify_hcm(caire_model(r_observed=false); treatment=:qa, outcome=:y).verdict  # :not_identified (bare IV graph)
```

Marking `r` observed (the mechanism constraint) is what buys identification; without it the engine
correctly returns `:not_identified`. Supplying that constraint, and choosing an estimator, are human
inputs the engine deliberately does not try to guess.

## Tests
Run `julia --project=. test/runtests.jl` (this package uses a direct test-file invocation,
not `Pkg.test()`, because this Julia setup cannot register the `Test` stdlib into the Pkg sandbox).
