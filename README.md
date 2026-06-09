# HCM.jl

Hierarchical causal models in Julia (Weinstein & Blei 2024), estimated with Turing.
A1 ships the **confounder** motif with a **hard + soft (stochastic) intervention** framework.

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
```

## Tests
Run `julia --project=. test/runtests.jl` (this package uses a direct test-file invocation,
not `Pkg.test()`, because this Julia setup cannot register the `Test` stdlib into the Pkg sandbox).
