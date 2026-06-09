# Design: HCM.jl — Sub-project A1 (foundation + hard/soft interventions + confounder motif)

- **Date:** 2026-06-09
- **Status:** Approved (brainstorming complete; pending implementation plan)
- **Location:** `~/projects/software/HCM.jl` (new Julia package; the R `hcm` and `CausalGraphs.jl` are untouched — A1 *vendors ideas* from R `hcm`; later Sub-project B *depends on* CausalGraphs.jl)
- **References:** Weinstein & Blei (2024) *Hierarchical Causal Models* (arXiv:2401.05330); Weinstein, Wood & Blei (2024) *Estimating the Causal Effects of T Cell Receptors* (CAIRE, arXiv:2410.14127) for the soft-intervention semantics; the R `hcm` package at `~/projects/software/hcm`.

## 0. Where A1 sits

HCM.jl is planned as: **Sub-project A** (Julia port of the motif estimators) + **Sub-project B** (a collapse-engine front-end to CausalGraphs.jl). A is split into **A1** (this spec) and **A2** (remaining motifs).

- **A1 (this spec):** package skeleton, spec/nested-graph representation, simulation, the **hard + soft intervention/estimand framework** (with CAIRE-ready extension points), and the **confounder** motif end-to-end in Turing, validated against ground truth (and parity-checked vs the R package).
- **A2 (later):** interference, nested, and the scalar **instrument** motif, reusing A1's patterns and intervention framework.
- **B (later):** `collapse`/`augment`/`marginalize` → flat ADMG → CausalGraphs.jl `identify`/`estimate_id`.

## 1. Purpose and scope

Provide an idiomatic-Julia, Turing-based implementation of hierarchical causal model estimation, beginning with the confounder motif, and establishing a **general intervention framework** supporting both hard interventions (`do(a = a★)`) and **soft/stochastic interventions** (`do(q^a ~ (1−ε)q^a + ε·δ_{a★})`, "add treatment `a★` at dosage `ε`"). The soft-intervention class is the capability the R package lacks and the one the CAIRE application centrally requires.

### In scope (A1)
- Julia package skeleton (`Project.toml`, module, tests, Documenter docs).
- `HCMSpec` resolution from a `@formula` + `unit`/`subunit` symbols + `motif`/`family`.
- `sim_hcm(:confounder; ...)` with analytic ground-truth ATEs for **both** intervention types.
- Intervention types `Hard(a★)` and `Soft(a★, ε)` as generic transforms of a within-unit treatment distribution.
- A pure `ate(fit, intervention; baseline)` estimand function (gaussian + bernoulli), computed post-hoc from posterior draws (no refit per intervention).
- The **full** confounder Turing model: both `q^a_i` (treatment propensity) and `q^{y|a}_i` (outcome law).
- `summary`, `compare_fe` (FE baseline via FixedEffectModels).
- Validation: deterministic estimand tests, recovery tests, recovery-parity vs R.

### Out of scope (A1)
- The interference, nested, and instrument motifs (→ A2).
- The collapse/identification engine (→ B).
- The CAIRE high-dimensional/neural backend (distributional treatments, IGoR, density-ratio fitness, deep-sets) — only **interface hooks** are designed, nothing is built.

## 2. Package identity, layout, API

**Dependencies:** `Turing`, `DataFrames`, `StatsModels`, `Distributions`, `Statistics`/`MCMCChains`, `FastGaussQuadrature` (quadrature for soft-intervention expectations where needed), `FixedEffectModels` (FE baseline); `Documenter` (docs).

**Layout:**
```
src/HCM.jl          # module + hcm() driver + motif dispatch
src/spec.jl         # HCMSpec: resolve formula/unit/subunit/motif/family
src/intervention.jl # Hard / Soft types + apply_intervention transform
src/sim.jl          # sim_hcm(:confounder; ...) with analytic true ATEs
src/models.jl       # confounder Turing @model
src/estimand.jl     # ate(draws, spec, intervention; baseline) -> ATE draws (pure)
src/methods.jl      # fit object, summary, quantiles
src/compare_fe.jl   # FixedEffectModels baseline
test/runtests.jl + test/test_*.jl
docs/               # Documenter
```

**API:**
```julia
fit = hcm(@formula(y ~ a), data; unit=:school, subunit=:student,
          motif=:confounder, family=:gaussian)   # family ∈ (:gaussian,:bernoulli)

ate(fit, Hard(1); baseline = Hard(0))   # hard ATE: E[Y;do(a=1)] − E[Y;do(a=0)]
ate(fit, Soft(1, 0.1))                  # soft: add a★=1 at dosage ε=0.1 vs ε=0
summary(fit)                            # posterior mean/sd/95% CI
compare_fe(fit)                         # FE baseline beside the hard ATE
```
`ate(fit, intervention; baseline)` returns a vector of ATE posterior draws, computed from the stored posterior without refitting. `baseline` defaults sensibly per intervention type (`Hard(0)` for `Hard`; the `ε=0` observed law for `Soft`).

## 3. Intervention / estimand framework

A within-unit treatment distribution `q^a_i` is transformed by an intervention:
- `Hard(a★)`: `q^a_i ↦ δ_{a★}`.
- `Soft(a★, ε)`: `q^a_i ↦ (1−ε)·q^a_i + ε·δ_{a★}`.

For the scalar binary confounder motif, `q^a_i = Bernoulli(p_i)` and the outcome law is `g⁻¹(β0_i + β1_i·a)` (`g⁻¹` = identity for gaussian, `logistic` for bernoulli). The estimand averages over units, per posterior draw:

- **Hard** `ate(Hard(1); baseline=Hard(0))` = `mean_i[ g⁻¹(β0_i+β1_i) − g⁻¹(β0_i) ]` (identical to the R confounder estimand).
- **Soft** `ate(Soft(a★, ε))` = `E[Y;do(σ_{a★,ε})] − E[Y;do(σ_{a★,0})]`
  `= ε · mean_i[ g⁻¹(β0_i + β1_i·a★) − ( p_i·g⁻¹(β0_i+β1_i) + (1−p_i)·g⁻¹(β0_i) ) ]`.
  - gaussian/identity simplifies to `ε · mean_i[ β1_i·(a★ − p_i) ]`.
  - bernoulli/logit uses the `logistic` mixture form above.
  The soft ATE is **linear in `ε`**, so one fit serves all dosages.

**Design implication (load-bearing):** the soft estimand needs `p_i` (within-unit treatment propensity), so the confounder model must estimate **both** `q^a_i` and `q^{y|a}_i` — unlike the R package, which modeled only the outcome law. A1 implements the full Eq. 18 model.

**CAIRE-ready hooks:** an intervention is implemented as `apply_intervention(intervention, q_unit) -> q_unit★` on a treatment-distribution object, and `ate` computes the expectation of the outcome law over the intervened distribution. For the scalar motif `q_unit = Bernoulli(p_i)`. A future high-dimensional/distributional `q` (e.g., a repertoire) implements the same mixture transform, and a neural outcome representation slots into the same `ate` skeleton — nothing in the intervention/estimand interface hardcodes the scalar case.

## 4. The confounder Turing model

Non-centred parameterisation; both sub-models:

```julia
@model function confounder_model(y, a, unit, n_units, family)
    μ   ~ MvNormal(zeros(2), 5I)
    τ   ~ filldist(truncated(Normal(0,2); lower=0), 2)
    μp  ~ Normal(0, 2);  σp ~ truncated(Normal(0,2); lower=0)
    zβ  ~ filldist(Normal(0,1), 2, n_units)
    zp  ~ filldist(Normal(0,1), n_units)
    β        = μ .+ τ .* zβ                # 2 × n_units  (q^{y|a})
    logitp   = μp .+ σp .* zp              # n_units       (q^a)
    a .~ Bernoulli.(logistic.(logitp[unit]))
    η = β[1, unit] .+ β[2, unit] .* a
    if family == :gaussian
        σy ~ truncated(Normal(0,2); lower=0);  y .~ Normal.(η, σy)
    else
        y .~ BernoulliLogit.(η)
    end
end
```
Draws extracted for the estimand: `β0_i = β[1,i]`, `β1_i = β[2,i]` (draws × n_units), `p_i = logistic(logitp_i)`.

## 5. Simulation

`sim_hcm(:confounder; n, m, params, family, seed)`:
- `u_i ~ Normal(0, sd_u)` (confounder).
- `p_i = logistic(α_p + ρ·u_i)` (propensity correlated with `u` → real confounding); `a_ij ~ Bernoulli(p_i)`.
- `y_ij = link(β0 + u_i + β1·a_ij)` (+ Normal noise for gaussian).
- Returns the data, the realized `u_i`/`p_i`, and **analytic ground truth**:
  - hard ATE: gaussian `= β1`; bernoulli `= mean_i[g⁻¹(β0+u_i+β1) − g⁻¹(β0+u_i)]`.
  - soft ATE`(a★,ε)`: gaussian `= ε·mean_i[β1·(a★−p_i)]`; bernoulli the `logistic` mixture form.

## 6. Validation (priority order)

1. **Deterministic estimand tests** (pure, no MCMC) — port the R discipline: hard ATE (gaussian contrast; bernoulli vs hand reference) AND soft ATE (gaussian `ε·mean[β1(a★−p)]`; bernoulli mixture vs hand reference). Catch port/Jensen/mixture bugs without sampling.
2. **Recovery** (Turing fit on sim): hard-ATE posterior 95% CI covers the true hard ATE; soft-ATE (e.g. `ε=0.2`) CI covers the true soft ATE; means within tolerance; no divergences.
3. **`compare_fe`**: FE baseline via `FixedEffectModels.reg(data, @formula(y ~ a + fe(unit)))` beside the hard ATE.
4. **Faithful-port check (recovery parity, primary):** HCM.jl recovers the same ground truth the R `hcm` recovers on a matched DGP. A same-dataset R-vs-Julia posterior-mean agreement script (tolerance ~0.05) is **optional** (`data-raw/`-style, not CI) — Stan-NUTS vs Turing-NUTS won't match bit-for-bit, so "faithful" means *both recover the same ground truth*, not identical posteriors.
5. **Hygiene:** `Pkg.test()` green; Documenter builds.

## 7. Open items for planning

- Exact draw-extraction API from the Turing/MCMCChains object into the `draws × n_units` matrices the estimand expects.
- Whether `apply_intervention` returns a distribution object or a closure for the outcome expectation (pick one and make it explicit in the plan).
- Prior defaults and how users override them through `hcm(...; kwargs)`.
- Whether the optional R-parity script shells out via `Rscript` or compares against a committed fixture of R outputs.
- Turing sampler defaults (NUTS, `adapt_delta`, chains/iter) and how they're surfaced.
