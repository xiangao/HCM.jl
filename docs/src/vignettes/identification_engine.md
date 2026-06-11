# General Identification Engine

The four motifs are worked examples, not a fixed menu. [`identify_hcm`](@ref) applies the
paper's identification procedure — *collapse* the plated graph to a flat model over
distribution-valued $Q$ variables, *augment*/*marginalize* as needed, then run do-calculus — to
**any** HCM graph, by latent-projecting to an ADMG and routing to
[CausalGraphs.jl](https://github.com/xiangao/CausalGraphs.jl).

Everything on this page is a pure graph computation: instant, no MCMC.

```
HCMGraph ──collapse()──▶ CollapsedModel ──augment()/marginalize()──▶
    ──latent_project()──▶ CausalGraphs.ADMG ──identify()──▶ verdict + functional
```

## The pipeline, step by step

Take the **confounder** graph and collapse it. Each subunit variable becomes a unit-level
$Q$ variable; the subunit→subunit edge `A → Y` is absorbed into the conditional *label*
`Q^{y|a}`, not a $Q\to Q$ edge:

```@example engine
using HCM
g = hcm_graph(vertices=[:U, :A, :Y], subunit=[:A, :Y], hidden=[:U],
              di_edges=[(:U, :A), (:U, :Y), (:A, :Y)])
m = collapse(g)
(vertices = m.vertices, edges = m.directed_edges, label_QY = m.labels[:QY])
```

The estimand is the effect of the treatment distribution on the outcome *marginal* $q^y$, so we
augment that node and run the whole pipeline:

```@example engine
r = identify_hcm(g; treatment=:QA, outcome=:qy,
                 augments=[(node=:qy, parents=[:QA, :QY])])
(verdict = r.verdict, motif = r.motif,
 admg_directed = r.admg.directed_edges, admg_bidirected = r.admg.bidirected_edges)
```

Latent-projecting out `U` leaves a bidirected edge `QA ↔ QY` (their shared hidden cause), and
do-calculus returns **`:a_fixable`** — a backdoor adjustment on `Q^{y|a}`, the paper's Eq. 25.

## The three motifs round-trip

Each built-in motif graph projects to the ADMG whose identification the paper derives by hand:

```@example engine
conf   = hcm_confounder()
interf = hcm_interference()
instr  = hcm_instrument()

verdict(spec) = identify_hcm(spec.graph; treatment=spec.treatment, outcome=spec.outcome,
                             augments=get(spec, :augments, []),
                             marginalizes=get(spec, :marginalizes, Symbol[])).verdict

(confounder   = verdict(conf),     # backdoor   (a-fixable)
 interference = verdict(interf),   # front-door (p-fixable)
 instrument   = verdict(instr))    # backdoor on q^{a|z}, after marginalizing q^z
```

## CAIRE: where graph machinery stops, and a domain assumption takes over

The framework's flagship application — Weinstein, Wood & Blei (2024),
[*Estimating the Causal Effects of T Cell Receptors*](https://arxiv.org/abs/2410.14127) — is the
instrument motif. Units are patients, subunits are T-cell-receptor sequences; the instrument is
the pre-selection repertoire. Identification turns on a population-genetics **selection law**,
$q^a = r \cdot q^z / \sum r\,q^z$, which reconstructs the latent fitness $r$ from the two observed
repertoire distributions — so $r$ becomes *effectively observed*. In the engine that is a
one-word change:

```@example engine
identify_hcm(caire_model(r_observed=true); treatment=:qa, outcome=:y).verdict
```

`:a_fixable` — backdoor adjustment on the reconstructed fitness $r$, exactly the paper's
Theorem 1. Drop the selection law (leave $r$ hidden) and the same graph is the bare IV graph,
which is **not** nonparametrically identified:

```@example engine
identify_hcm(caire_model(r_observed=false); treatment=:qa, outcome=:y).verdict
```

## Why the engine is identification-only

CAIRE shows that an HCM application has three layers, and they generalize very differently:

| Layer | Generalizable? | Where the value is |
|---|---|---|
| collapse + do-calculus → verdict & functional | **yes** — a pure graph algorithm | low (CAIRE: one theorem) |
| mechanism constraints that make a latent observable (the selection law) | **no** — domain algebra | **high** — without it, not identified |
| estimation of the identified functional (CAIRE: density-ratio CNN, deep sets, amortized VI) | **no** in general — bespoke | **high** — most of an applied paper |

The engine automates the first layer. The mechanism constraint (mark-the-latent-observed) and the
estimator are explicit human inputs — which is why a fully general "any HCM in, an estimate out"
tool is not the goal, and why the engine correctly refuses CAIRE-without-its-assumption.
