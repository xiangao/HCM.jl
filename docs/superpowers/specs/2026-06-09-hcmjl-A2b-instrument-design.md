# Design: HCM.jl — Sub-project A2b (scalar instrument motif, hard + soft interventions)

- **Date:** 2026-06-09
- **Status:** Approved (brainstorming complete; pending implementation plan)
- **Location:** `~/projects/software/HCM.jl` (extends A1 + A2a)
- **References:** Weinstein & Blei (2024) HCM paper §4.3 / Fig. 2i-2j / Eq. 33-34 (instrument graph); Weinstein, Wood & Blei (2024) CAIRE/TCR paper (the flagship instrument application); A1/A2a specs+plans.

## 0. Where A2b sits

HCM.jl roadmap: **A** (motif port) = **A1** (confounder, DONE) + **A2** (= **A2a**, interference+nested, DONE; **A2b**, this spec) ; **B** (collapse engine on CausalGraphs.jl, next sub-project). A2b completes the four-motif set.

## 1. Purpose and scope

Add the **scalar instrument motif** to HCM.jl: a subunit-level instrument `Z` shifts a subunit treatment `A`, the outcome `Y` is **unit-level**, and a hidden unit confounder `U` affects both the treatment response and the outcome. This is the scalar, faithful version of the HCM instrument graph (and of the TCR/CAIRE structure, without its domain-specific selection mechanism). Both **hard and soft interventions** on the within-unit treatment rate `q^a`, via the established `Hard`/`Soft` framework.

### In scope (A2b)
- `InstrumentSpec` (unit-level outcome `y`; subunit `z`, `a`; `unit_id`).
- `instrument_model` Turing model (exogenous instrument `q^z`; confounded treatment response `q^{a|z}`; unit outcome on the induced rate `q^a` adjusting for `q^{a|z}`).
- `_ate(::Val{:instrument})` hard + soft estimand (gaussian + bernoulli), backdoor-adjusted on `q^{a|z}`.
- `sim_hcm(:instrument; ...)` with analytic hard/soft truth and a confounded DGP.
- `compare_fe` instrument branch: HCM backdoor `θ_a` vs naive OLS of `y` on `q^a`.
- Validation: deterministic estimand tests, NUTS recovery + naive-bias demonstration.

### Out of scope (A2b)
- The collapse/identification engine (→ B).
- The CAIRE high-dimensional/neural/selection-mechanism machinery.
- Classic IV LATE (the target is the HCM `do(q^a)` rate effect, not the complier effect).

## 2. The instrument graph and estimand (decided)

HCM instrument graph (Fig. 2i): subunit instrument `Z_ij` → subunit treatment `A_ij`; hidden unit confounder `U_i` → treatment response and → outcome; **unit-level** outcome `Y_i`. Collapsed (Fig. 2j), identification (Eq. 34) intervenes on the within-unit treatment distribution `q^a` and **backdoor-adjusts on the instrument-response `q^{a|z}`**, with the exogenous instrument distribution `q^z` supplying the identifying variation. The target estimand is the **HCM `do(q^a)` rate effect** (population effect of the treatment rate) — NOT a classic LATE.

## 3. Model (`instrument_model`)

Gaussian (bernoulli analogous); units `i`, subunits `j`:
- **Instrument (exogenous):** `z_ij ~ Bernoulli(ω_i)`, `ω_i` pooled, independent of `U` — `q^z`.
- **Treatment response (confounded `q^{a|z}`):** `a_ij ~ Bernoulli(logistic(α_i + β_i·z_ij))`, `(α_i, β_i)` pooled; `π0_i = logistic(α_i)`, `π1_i = logistic(α_i + β_i)`.
- **Induced treatment rate:** `q^a_i = ω_i·π1_i + (1−ω_i)·π0_i`.
- **Unit-level outcome (one per unit):** `y_i ~ Normal(θ0 + θ_a·q^a_i + θ_{r0}·π0_i + θ_{r1}·π1_i, σy)` (bernoulli: `BernoulliLogit` of the same linear predictor).

Conditioning `y` on `(π0_i, π1_i)` is the Eq. 34 backdoor adjustment (absorbs `U`'s confounding of `y`); the exogenous `ω_i` makes `q^a` vary beyond `(π0,π1)`, identifying `θ_a` (multi-site-IV with random coefficients). The model returns `(; θ0, θa, θr0, θr1, π0, π1, qa)` per draw (π0,π1,qa as draws×n_units).

## 4. Estimand (`_ate(::Val{:instrument})`)

`do(q^a = q*)` per draw, averaged over units:
`E[Y; do(q^a=q*)] = mean_i[ link_inv(θ0 + θ_a·q* + θ_{r0}·π0_i + θ_{r1}·π1_i, family) ]`.
- **Hard** `ate(Hard(1); baseline=Hard(0))`: gaussian `= θ_a`; bernoulli `= mean_i[link_inv(B_i+θ_a) − link_inv(B_i)]` with `B_i = θ0 + θ_{r0}·π0_i + θ_{r1}·π1_i`.
- **Soft** `ate(Soft(a★, ε))`: set the rate to `(1−ε)·q^a_obs_i + ε·a★`; `E[Y;do(σ_{a★,ε})] − E[Y;do(σ_{a★,0})]`; gaussian simplifies to `θ_a · ε · mean_i(a★ − q^a_obs_i)`. (`q^a_obs_i` = the unit's observed treatment rate, stored in the spec.)

Marginalizes over realized per-unit `(π0,π1)` (and `q^a_obs` for soft), consistent with the package's bernoulli-Jensen discipline.

## 5. Spec (`InstrumentSpec`)

Fields: `outcome, treatment, instrument` (symbols); `y::Vector` (length **n_units**, unit-level); `a::Vector{Int}`, `z::Vector{Int}` (length N, subunit); `unit_id::Vector{Int}` (length N); `n_units`; `qa_obs::Vector{Float64}` (per-unit observed treatment rate = `mean(a)` within unit); `family`, `motif`. Constructor `hcm_spec(@formula(y ~ a), data; unit=:unit, subunit=:subunit, instrument=:z, motif=:instrument, family=...)` — note `y` is taken as the unit-level value (one per unit; validate constant within unit, take first).

## 6. Simulation (`sim_hcm(:instrument; ...)`)

`u_i` confounder; `ω_i` exogenous (independent of `u`); `α_i = α0 + ρ·u_i + noise`, `β_i = β0 + noise` (response confounded via `α`); `z_ij ~ Bern(ω_i)`, `a_ij ~ Bern(logistic(α_i+β_i·z_ij))`; `q^a_i = ω_i·π1_i + (1−ω_i)·π0_i`; `y_i = θ0 + θ_a·q^a_i + ψ·u_i + noise` (gaussian) / `BernoulliLogit` (bernoulli). Returns `data` (cols `unit, subunit, z, a` at subunit level and `y` repeated per row at unit level), `true_hard = θ_a` (gaussian) / marginalized (bernoulli), `true_soft(a★,ε)`, and realized `u, ω, π0, π1, qa`. Because `q^a` correlates with `u`, a naive `y~q^a` regression is biased by `ψ`; the backdoor adjustment recovers `θ_a`.

## 7. Reporting / comparison

`compare_fe` gains an `:instrument` branch returning a 2-row table: "HCM (backdoor θ_a)" = `mean(ate(fit, Hard(1); baseline=Hard(0)))`, and "Naive OLS (y ~ q^a)" = the slope from `reg(unit_df, @formula(y ~ qa))` on the per-unit `(y_i, q^a_obs_i)` (unit-FE does not apply to a one-per-unit outcome). The naive slope is biased; the HCM `θ_a` is not.

## 8. Validation

1. **Deterministic estimand tests (primary):** gaussian hard `= θ_a`; gaussian soft `= θ_a·ε·mean(a★−q^a_obs)`; bernoulli hard/soft vs hand reference. Pure, no MCMC.
2. **Recovery + naive-bias demonstration (NUTS):** the backdoor `θ_a` CI covers `true_hard`; the naive OLS slope is materially further from `θ_a` than the HCM estimate (the value demonstration). One run.
3. **`compare_fe` instrument branch** returns the two-row table.
4. **Regression guard:** A1/A2a tests still pass (A2b is additive: new `Val(:instrument)` estimand methods + `InstrumentSpec` + `instrument_model` + driver branch).
5. Full `julia --project=. test/runtests.jl` once at the end.

## 9. Plan decomposition (~5 tasks)

1. **instrument sim** (`sim_hcm(:instrument)` + analytic hard/soft truth; test).
2. **instrument spec** (`InstrumentSpec`, unit-level `y`, `qa_obs`; test).
3. **instrument estimand** (`_ate(::Val{:instrument})` hard+soft; deterministic tests).
4. **instrument model + driver + recovery** (`instrument_model`, driver branch, recovery + naive-bias demonstration test).
5. **`compare_fe` instrument branch + docs + full suite.**

## 10. Open items for planning

- Exact `draws` extraction (model returns `θ0,θa,θr0,θr1, π0,π1, qa`; driver builds per-unit matrices like A2a).
- Whether `ω_i` is per-unit pooled or a single shared instrument propensity (default: per-unit pooled, independent of `u`).
- Identifiability note: `θ_a` is identified only if `ω_i` varies across units (gives `q^a` variation beyond `(π0,π1)`); the sim must induce `ω` spread. Document; the recovery test's DGP ensures it.
- Bernoulli soft-estimand baseline handling (`Soft(a★,0)` uses observed rate) — mirror the confounder/nested soft form.
- `y` unit-level extraction in the spec (validate constant within unit).
