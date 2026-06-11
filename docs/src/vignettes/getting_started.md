# Getting Started

This vignette fits each of the four motifs and reads off an average treatment effect. Every fit is
hierarchical-Bayes via Turing/NUTS — minutes per fit — so the code below is **runnable as written**
but not executed during the docs build. In every case `sim_hcm` returns the analytic ground truth
`true_hard`, so you can check the posterior against it; we describe what each estimator targets
rather than pasting sampler-specific numbers. (For executed, instant examples see the
[General Identification Engine](identification_engine.md) and [Reference](../reference.md) pages.)

## Confounder

Unit-level confounding only (the [`sim_hcm`](@ref) `:confounder` DGP has a hidden `u` driving both
the propensity and the outcome). The within-unit backdoor identifies the effect; in the linear
case it coincides with fixed effects, which [`compare_fe`](@ref) reports.

```julia
sim = sim_hcm(:confounder; n=60, m=40, seed=1)
fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit, motif=:confounder)

summarize_ate(ate(fit, Hard(1); baseline=Hard(0)))   # hard do(a=1) vs do(a=0)
sim.true_hard                                         # ground truth (β1 = 0.5)
compare_fe(fit)                                       # fixed-effects baseline
```

The posterior ATE recovers `sim.true_hard` (here `0.5`), and `compare_fe` returns essentially the
same number: in the linear-Gaussian confounder model the HCM estimate **is** the within-unit
fixed-effects slope.

A **soft** intervention adds `a★` to a fraction `ε` of subunits; the gaussian soft ATE is linear in
`ε` (≈ `ε · mean_i β1ᵢ(a★ − pᵢ)`), so dosing 10% gives roughly a tenth of the hard effect:

```julia
summarize_ate(ate(fit, Soft(1, 0.1)))                 # dose 10% of subunits to a=1
```

## Confounder & interference

Treatment acts through a unit-level channel `z` (with unit covariate `s`) that feeds back onto
every subunit's outcome. The hard ATE is the **full-do total effect** — the do is propagated
through the channel, not just the direct term:

```julia
sim = sim_hcm(:confounder_interference; n=60, m=40, seed=1)
fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit,
          motif=:confounder_interference, interferer=:z, unit_covar=:s)

summarize_ate(ate(fit, Hard(1); baseline=Hard(0)))    # total effect, incl. spillover
sim.true_hard
```

The posterior recovers `sim.true_hard`, the **total** effect `μ1 + (δ0+δ1)·γ1` — direct plus the
spillover routed through the channel. A confounder-motif fit on the same data would instead recover
only the direct effect `μ1 + δ1·E[z]`; the [Interference vs. SUTVA](interference_vs_sutva.md)
vignette quantifies that gap and shows it growing with the interference strength.

## Nested confounder (three levels)

Students in classes in schools, with confounding at both aggregate levels. The identified ATE is
the within-class effect; [`nested_diagnostics`](@ref) returns the Mundlak group-mean coefficients
as a confounding diagnostic.

```julia
sim = sim_hcm(:nested_confounder; n=20, m=30, seed=1)   # n schools, m students/class
fit = hcm(@formula(y ~ a), sim.data; groups=(:school, :class), motif=:nested_confounder)

summarize_ate(ate(fit, Hard(1); baseline=Hard(0)))      # within-class effect, true β1 = 0.5
nested_diagnostics(fit.draws)                           # λclass, λschool ≈ confounding signal
```

The posterior ATE recovers the within-class effect (`sim.true_hard = 0.5`). The Mundlak group-mean
coefficients `λclass`, `λschool` from `nested_diagnostics` sit away from zero when between-group
variation is confounded — a Hausman-style flag that you were right to identify off the within-class
contrast.

## Instrument (subunit IV, unit outcome)

A subunit instrument `z` shifts treatment within each unit; the outcome `y` is unit-level. The
effect of the treatment rate is identified by a backdoor adjustment on the
instrument-conditional propensity `q^{a|z}`.

```julia
sim = sim_hcm(:instrument; n=80, m=40, seed=1)
fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit,
          instrument=:z, motif=:instrument)

summarize_ate(ate(fit, Hard(1); baseline=Hard(0)))      # treatment-rate effect θ_a, true 0.5
sim.true_hard                                           # θ_a = 0.5
compare_fe(fit)                                         # naive (biased) OLS of y on q^a
```

The HCM posterior for the treatment-rate effect `θ_a` is centred near the truth (`0.5`) with a wide
interval (see the caveat below), while `compare_fe`'s naive OLS of the unit outcome on the observed
rate `q^a` is biased upward because it omits the `q^{a|z}` adjustment that blocks the confounder.

!!! note "Weak-instrument caveat"
    The realized treatment rate is highly collinear with the propensity adjustment set, so only
    the instrument-driven residual identifies `θ_a`. The posterior interval reliably **covers** the
    truth (the robust-IV guarantee), but the point estimate is imprecise unless the instrument is
    strong.
