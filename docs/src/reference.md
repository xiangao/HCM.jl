# Reference

## Fitting and estimands

```@docs
hcm
ate
summarize_ate
compare_fe
nested_diagnostics
```

## Interventions

```@docs
Hard
Soft
```

## Simulation

```@docs
sim_hcm
```

## General identification engine

```@docs
hcm_graph
collapse
augment
marginalize
latent_project
identify_hcm
collapsed_model
```

### Built-in graphs

`hcm_confounder`, `hcm_interference`, `hcm_instrument` return the three motif graphs (as
NamedTuples carrying the `graph`, `treatment`, `outcome`, and any `augments`/`marginalizes`);
`caire_model(; r_observed)` returns the CAIRE collapsed repertoire-IV model. These are the inputs
used throughout the [General Identification Engine](vignettes/identification_engine.md) vignette.

```@docs
hcm_confounder
hcm_interference
hcm_instrument
caire_model
```

```@example ref
using HCM
g = hcm_confounder().graph
collapse(g).labels         # the Q-variable labels produced by collapsing
```
