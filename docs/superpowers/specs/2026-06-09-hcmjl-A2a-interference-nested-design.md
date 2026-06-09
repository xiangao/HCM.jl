# Design: HCM.jl — Sub-project A2a (interference + nested motifs, hard + soft interventions)

- **Date:** 2026-06-09
- **Status:** Approved (brainstorming complete; pending implementation plan)
- **Location:** `~/projects/software/HCM.jl` (extends A1; the R `hcm` package at `~/projects/software/hcm` is the porting source, untouched)
- **References:** A1 spec/plan (`docs/superpowers/specs|plans/2026-06-09-hcmjl-A1*`); R `hcm` `R/estimand.R`, `inst/stan/confounder_interference.stan`, `inst/stan/nested_confounder.stan`; Weinstein & Blei (2024) HCM paper (Eq. 19/42 interference, Mundlak nested).

## 0. Where A2a sits

HCM.jl roadmap: **A** (motif port) = **A1** (confounder + hard/soft, DONE) + **A2** (remaining motifs); **B** (collapse engine on CausalGraphs.jl). A2 splits into **A2a** (this spec: interference + nested ports, hard+soft) and **A2b** (scalar instrument motif, later).

## 1. Purpose and scope

Port the **interference** (`confounder_interference`) and **nested** (`nested_confounder`) motifs from R `hcm` into HCM.jl/Turing, each extended to **hard and soft (stochastic) interventions** via A1's `Hard`/`Soft` framework. Soft support is the value-add over the hard-only R package.

### In scope (A2a)
- A small refactor of A1's estimand engine to dispatch per motif (Approach A; see §2).
- Two Turing models (`interference_model`, `nested_model`), each modeling the outcome law, the interferer/Mundlak structure, AND a treatment-propensity sub-model `q^a` (required for soft).
- Per-motif hard + soft estimands (gaussian + bernoulli), as pure dispatched functions.
- `sim_hcm` branches with analytic hard + soft ground truth for both motifs.
- Spec resolution for the interferer + unit covariate (interference) and the group nesting (nested).
- Per-motif `compare_fe` and reporting (nested `λ` diagnostics + variance components; interference front-door coefficients).
- Validation: deterministic estimand tests (primary), NUTS recovery (confirmatory), A1-regression guard.

### Out of scope (A2a)
- The scalar instrument motif (→ A2b).
- The collapse/identification engine (→ B).
- High-dimensional/neural (CAIRE) backends.

## 2. Architecture — per-motif estimand dispatch (Approach A)

Generalize A1's estimand engine. `ate(fit, intv; baseline)` calls
`expected_outcome(Val(fit.spec.motif), fit.draws, fit.spec, intv)`.

`HCMFit` gains a generic `draws::NamedTuple` field holding the per-motif posterior draws:
- confounder: `(β0, β1, p)` — A1's matrices, **moved into `draws`**; A1's confounder `expected_outcome` rewritten to read `draws.β0` etc. (the only change to existing code; A1 confounder tests must stay green).
- interference: `(μ0, μ1, δ0, δ1, γ0, γ1, γ2, σz, u, p)` — population params (draws-vectors), per-unit intercept `u` (draws×n_units), per-unit propensity `p` (draws×n_units).
- nested: `(β0, β1, λclass, λschool, b_school, b_class, p_class)` — Mundlak params (draws-vectors), nested intercepts (draws×K, draws×C), per-class propensity (draws×C).

Each motif defines `expected_outcome(::Val{:motif}, draws, spec, ::Hard)` and `(::Soft)` — focused, independently testable. Shared `Hard`/`Soft` types and the single `ate(fit, intv)` surface are unchanged. This is the extension path the A1 review identified; A2b's instrument motif and a future distributional backend plug in identically.

**Both ported models gain a `q^a` treatment-propensity sub-model** because soft interventions need the within-unit/within-class propensity `p` — the recurring "soft needs `q^a`" lesson from A1. The Julia models are thus the R models plus a propensity sub-model.

## 3. API additions

```julia
# interference
hcm(@formula(y ~ a), data; unit=:cell, subunit=:factory,
    motif=:confounder_interference, interferer=:enforcement, unit_covar=:s)
# nested
hcm(@formula(y ~ a), data; groups=(:school, :class), motif=:nested_confounder)
```
`interferer`/`unit_covar` are column symbols (unit-level; validated constant within unit). `groups` is a tuple of symbols, coarsest-first (Julia analogue of R `~ school/class`; class nesting via interaction).

## 4. Models and estimands

### 4.1 Interference (`interference_model`)
Generative (gaussian; bernoulli analogous):
- `u_i ~ Normal(0, τ0)` (per-unit confounder intercept).
- `a_ij ~ Bernoulli(logistic(πlogit_i))`, `πlogit_i` pooled — the `q^a` sub-model (new).
- `z_i ~ Normal(γ0 + γ1·ā_i + γ2·s_i, σz)` (interferer mechanism; `ā_i` = within-unit treatment share; `z` observed, unit-level).
- `y_ij ~ link(μ0 + u_i + δ0·z_i + (μ1 + δ1·z_i)·a_ij)`.

Estimands (per draw; GH quadrature over `z`; marginalize over realized `u_i`):
- **Hard** `do(a=a★)`: `E_i^{do(a★)} = ∫ link_inv(μ0+u_i+δ0·z+(μ1+δ1·z)·a★)·N(z; γ0+γ1·a★+γ2·s_i, σz) dz`; `ATE = mean_i[E_i^{do(1)} − E_i^{do(0)}]`.
- **Soft** `do(σ_{a★,ε})`: `ā_new_i = (1−ε)·ā_obs_i + ε·a★`; outcome expectation mixes
  `(1−ε)[p_i·m_y(1,z) + (1−p_i)·m_y(0,z)] + ε·m_y(a★,z)` with `m_y(a,z)=link_inv(μ0+u_i+δ0·z+(μ1+δ1·z)·a)`, integrated over `z ~ N(γ0+γ1·ā_new_i+γ2·s_i, σz)`; `ATE = mean_i[E_i^{do(σ_{a★,ε})} − E_i^{do(σ_{a★,0})}]` (baseline uses `ā_obs` and the observed a-mixture).

### 4.2 Nested (`nested_model`)
Generative:
- `η_ij = β0 + β1·a_ij + λclass·ābar_class[c] + λschool·ābar_school[k] + b_school[k] + b_class[c]`; `b_school ~ Normal(0,σschool)`, `b_class ~ Normal(0,σclass)`.
- `a_ij ~ Bernoulli(logistic(πlogit_class[c]))`, per-class propensity pooled — `q^a` sub-model (new).
- `y_ij ~ link(η_ij)`.
- Per-student baseline `B_i = β0 + λclass·ābar_class + λschool·ābar_school + b_school + b_class`.

Estimands (per draw; marginalize over realized `b_school,b_class`):
- **Hard** within-class ATE: gaussian `= β1`; bernoulli `= mean_i[link_inv(B_i+β1) − link_inv(B_i)]`.
- **Soft** `do(σ_{a★,ε})`: `ε · mean_i[ link_inv(B_i+β1·a★) − (p_{c(i)}·link_inv(B_i+β1) + (1−p_{c(i)})·link_inv(B_i)) ]`; gaussian simplifies to `ε·mean_i[β1·(a★ − p_{c(i)})]`.

## 5. Simulation

`sim_hcm` branches (port R DGPs; return analytic hard + soft truth):
- `:confounder_interference`: `u_i`; `p_i`; `a_ij ~ Bernoulli(p_i)`; `z_i ~ N(γ0+γ1·ā_i+γ2·s_i, σz)`; `y ~ link(μ0+u_i+δ0·z+(μ1+δ1·z)·a)`. `true_hard`, `true_soft(a★,ε)` = the §4.1 formulas at true params with realized `u_i, p_i, s_i`.
- `:nested_confounder`: school+class confounders, per-class propensity correlated with them, Mundlak outcome. `true_hard = β1` (gaussian); `true_soft(a★,ε) = ε·mean[β1·(a★−p_class)]`.

## 6. Reporting and FE baselines

- `summarize_ate` unchanged. For nested, `summary`/diagnostics surface `λclass`, `λschool` (Hausman confounding diagnostic) and `σschool`, `σclass` (variance decomposition); for interference, the front-door coefficients (`δ`, `γ`).
- `compare_fe` dispatches per motif: nested → two-way FE `reg(df, @formula(y ~ a + fe(school) + fe(class)))`; interference → the matching within/FE baseline (`y ~ a + fe(unit)`), reported beside the HCM hard ATE.

## 7. Validation

1. **Deterministic estimand tests (primary):** per motif × {hard, soft} × {gaussian, bernoulli}. Gaussian-soft simplifications exact; bernoulli and the **interference soft dose-through-channel** vs a brute-force MC reference. These catch front-door/Jensen errors without MCMC.
2. **NUTS recovery (confirmatory, one run per motif):** hard *and* soft ATE 95% CI covers the analytic truth.
3. **`compare_fe`** per motif returns the matching FE estimate beside the HCM ATE.
4. **A1-regression guard:** the A1 confounder estimand/recovery tests still pass after the `draws`-refactor.
5. Full `julia --project=. test/runtests.jl` once at the end (NUTS recovery is ~30 min/motif; per-motif iteration uses the fast deterministic tests).

## 8. Plan decomposition (sequencing)

1. **Estimand-dispatch refactor:** `Val(motif)` + `draws::NamedTuple`; rewrite confounder `expected_outcome` to read `draws`; confounder tests green.
2. **Nested motif** (simpler, establishes the multi-motif pattern): sim → `nested_model` → hard+soft estimand (deterministic tests) → recovery → reporting/`compare_fe`.
3. **Interference motif** (intricate soft front-door): sim → `interference_model` → hard+soft estimand (deterministic + MC tests) → recovery → reporting/`compare_fe`.
4. **Docs + single full-suite run.**

## 9. Open items for planning

- Exact `draws::NamedTuple` construction in the `hcm()` driver per motif (extend the A1 `generated_quantities` extraction; interference/nested models `return` the needed transformed quantities).
- GH-quadrature node count for the interference `z` integral (reuse A1's choice; add `FastGaussQuadrature` as a dep — now actually needed).
- Nesting resolution in Julia spec (interaction of group symbols, coarsest-first) and the per-class/per-unit propensity index plumbing.
- Whether the soft-interference MC reference test uses a fixed-seed Monte Carlo tolerance (~1e-3) as in the R deterministic bernoulli test.
- Sizes for the (slow) recovery tests to keep them tractable while still covering both hard and soft.
