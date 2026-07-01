# HCM.jl — Project Notes

## Overview

Julia implementation of Weinstein & Blei (2024), *Hierarchical Causal Models*
(arXiv:2401.05330), estimated with Turing.jl. An HCM has two levels — *units* (schools,
regions, patients) and, nested inside each, *subunits* (students, factories, cells).
Treatment and outcome can live at either level. The paper's key move: when treatment is
measured at the subunit level, a unit-level confounder is held fixed *within* each unit,
so the within-unit subunit data behaves like a natural experiment — collapse the plated
graph to a flat model over distribution-valued `Q` variables, then run do-calculus.

This is well past the original A1/A2a/A2b roadmap — treat the version below as current,
not the plan docs under `docs/superpowers/`, which describe earlier design intent.

## What's implemented (verify by reading `src/`, not this file, if in doubt)

**Four estimation motifs**, each with hard (`Hard(a*)`) and soft/stochastic
(`Soft(a*, ε)`) interventions, dispatched in `src/HCM.jl::hcm()` by `motif::Symbol` and
in `src/estimand.jl::_ate` by `Val(motif)`:

- `:confounder` — unit-level hidden confounder; within-unit backdoor (linear case =
  fixed effects). The original A1 motif.
- `:confounder_interference` — treatment acts through a unit-level channel `z`
  (front-door, full-do propagated through the channel). A2a.
- `:nested_confounder` — three-level nesting (e.g. students/classes/schools) via
  Mundlak-style correlated random effects (`groups=(:school,:class)`). A2a.
- `:instrument` — subunit instrument `z` → treatment, unit-level outcome; backdoor on
  `q^{a|z}`. This is **A2b, and it is done**, not deferred — see `src/models.jl`'s
  `instrument_model`, `src/estimand.jl`'s instrument `_ate`, and
  `test/test_instrument*.jl`. Its estimand `θ_a` is explicitly a weak-instrument
  estimand: the 95% posterior interval reliably covers the truth, but the point
  estimate is only precise when the instrument has strong within-unit spread — this
  is asserted as a coverage check, not a point-estimate check, in the recovery tests.

Two more motifs beyond the original plan, added for panel/DiD applications:
- `:did` and `:did_interference` — joint HCM + difference-in-differences (panel data,
  `groups=(:school, :period)`); `did_direct()` in `estimand.jl` separates the direct
  from the total (spillover-propagated) effect for `:did_interference`.

**A general identification engine** (`src/collapse.jl`, ~386 lines) — this was the
"collapse engine on CausalGraphs.jl" item and **it is also done**, not deferred.
Pipeline: `HCMGraph` (plate-aware DAG with hidden nodes) → `collapse()` → `augment()` /
`marginalize()` → `latent_project()` → `CausalGraphs.ADMG` → `identify()`, returning a
verdict (`:a_fixable` backdoor, `:p_fixable` front-door, `:nested_fixable`,
`:id_algorithm`, or `:not_identified`) plus the identified functional. `identify_hcm()`
is the entry point; `hcm_confounder()`, `hcm_interference()`, `hcm_instrument()`, and
`caire_model()` are built-in graphs that round-trip to the corresponding motif. This
engine is deliberately **identification-only** — see the "three layers" discussion in
`README.md` and `docs/src/index.md`: it automates graph → verdict + functional, but
mechanism constraints that make a latent observable (e.g. CAIRE's population-genetics
selection law) and estimation of the identified functional are human/domain inputs it
does not try to guess.

**Nothing from the original roadmap is currently listed as deferred** in the source or
README. If you're picking up new work, check `git log --oneline` for the actual frontier
rather than assuming A2b or the collapse engine still need building.

## Architecture

- `src/collapse.jl` — `HCMGraph`, `collapse`/`augment`/`marginalize`/`latent_project`,
  `identify_hcm`, `recognize_motif`, built-in graphs, `to_mermaid`.
- `src/intervention.jl` — `Hard`/`Soft` intervention structs (16 lines, trivial).
- `src/estimand.jl` — `ate()`, `link_inv()` (gaussian/bernoulli), `did_direct()`; per-motif
  `_ate(::Val{motif}, ...)` dispatch.
- `src/sim.jl` — `sim_hcm(motif; ...)` DGP simulators with analytic hard/soft truth for
  every motif; the source of truth used by every recovery test.
- `src/spec.jl` — `hcm_spec()` + per-motif spec structs (`HCMSpec`, `NestedSpec`,
  `InterferenceSpec`, `InstrumentSpec`, `DiDSpec`, `DiDInterferenceSpec`) that turn a
  `@formula` + long-format DataFrame into the arrays each Turing model consumes.
- `src/models.jl` — the `@model` Turing definitions, one per motif.
- `src/methods.jl` — `hcm()` dispatch, `HCMFit`, `summarize_ate`, `nested_diagnostics`.
- `src/compare_fe.jl` — classical FE/OLS baseline for each motif, to show where HCM and
  the naive/SUTVA estimate agree or diverge (e.g. instrument's naive OLS on `q^a` vs
  HCM's backdoor `θ_a`).

**Key invariant**: every motif's Turing model is written with explicit `for`-loops over
observations, not `.~` broadcast-tilde — see the in-code comment at `src/models.jl:149`
("DynamicPPL ≥ v0.35: arrays of distributions must use loops, not `.~`"). Don't
"simplify" a loop back to `.~` — it will look correct and then silently misbehave/error
under the pinned DynamicPPL version (check `Manifest.toml` for the exact version if this
ever needs revisiting).

**Sampling**: all motifs call `sample(model, NUTS(...), MCMCSerial(), iter, chains; ...)`
(see every branch of `hcm()` in `src/HCM.jl`). This machine's Julia process runs with 1
thread, so multi-chain sampling must go through `MCMCSerial()`, not `MCMCThreads()`. The
`adtype` kwarg defaults to `AutoForwardDiff()`; pass `AutoReverseDiff()` etc. for
reverse-mode AD on many-parameter models.

## Companion package: R `hcm` (lowercase, different repo)

`~/projects/software/hcm` is a **separate, from-scratch** implementation of the same
Weinstein-Blei framework, using Stan/cmdstanr instead of Turing.jl. It currently covers
only the confounder and confounder+interference motifs (per its `DESCRIPTION`), so it is
behind this Julia package (no nested, instrument, DiD, or identification-engine
equivalent there as of this writing). Treat the two as independent implementations for
cross-checking numerically (same DGP, same paper, different PPL backend) — do not assume
a fix or feature in one carries over to the other; port deliberately if you want parity.

## Tests

**Do not run the full suite casually.** Read `test/runtests.jl` before running anything.

The default invocation is now fast (~35s) because the slow NUTS posterior-recovery
testsets are gated behind an environment variable and skipped otherwise:

```bash
julia --project=. test/runtests.jl                    # fast: recovery tests are @test_skip'd
HCM_SAMPLING_TESTS=1 julia --project=. test/runtests.jl  # slow: runs real NUTS fits (this is the ~40 min path)
```

The gate (`get(ENV, "HCM_SAMPLING_TESTS", "") != "1"` → `@test_skip`) lives at the top of
`test_confounder.jl`, `test_nested.jl`, `test_interference.jl`, `test_instrument.jl`, and
`test_compare_fe.jl`. This gating was added deliberately (commit `9f6ac1c`) to replace an
earlier serial recovery suite that took ~138 minutes.

For a faster full-coverage recovery check, use `test/recovery_parallel.jl` instead of
`HCM_SAMPLING_TESTS=1`: it runs all six motifs' NUTS recoveries concurrently via
`Distributed`/`pmap` (capped at `min(6, Sys.CPU_THREADS - 1)` workers — this machine is
6 physical cores, so this is the right cap already), each asserting 95% posterior
coverage of the simulated truth:

```bash
julia --project=. test/recovery_parallel.jl
```

**Always invoke tests via the direct file path**, never `Pkg.test()`:

```bash
julia --project=. test/runtests.jl
```

`Pkg.test()` breaks on this package under the current Julia 1.11/1.12 setup because the
Pkg sandbox it spawns cannot register the `Test` stdlib (documented in `README.md`'s
Tests section). This is a project-Manifest/sandbox quirk, not a code bug — don't spend
time trying to "fix" it into working with `Pkg.test()`.

## CI: docs only, deliberately no test workflow

`.github/workflows/docs.yml` builds and deploys the Documenter site on push to `master`
(via `Pkg.develop` + `docs/make.jl`). There is **no CI job that runs the test suite**.
This is deliberate, not an oversight: the full recovery suite is ~40 minutes and even the
parallel recovery runner fits six real NUTS models, too slow/expensive for routine CI on
every push. If CI test coverage is ever wanted, don't just bolt on the standard
`julia-actions/julia-runtest` template — it would either run the fast (`@test_skip`'d,
so meaningless for correctness) suite or the full slow one. Instead, design a genuine
smoke-test subset first (e.g. `:confounder` only, `iter=100, chains=1`, asserting the
`hcm()`/`ate()` call succeeds and returns finite draws, not full posterior coverage).

## Docs

`docs/src/index.md` + `docs/src/vignettes/` (getting_started, identification_engine,
interference_vs_sutva) are the up-to-date narrative overview — closer to current state
than `docs/superpowers/plans/` and `docs/superpowers/specs/`, which are historical design
docs from the A1/A2a/A2b build-out (2026-06-09) and don't reflect later additions like
`:did`/`:did_interference` or the identification engine's completion. Rendered site:
https://xiangao.github.io/HCM.jl/.

## Misc

- `examples/` has standalone scripts (`interference_showcase.jl`,
  `interference_confirm.jl`, `capture_all.jl`) with their own `Project.toml`/`Manifest.toml`.
- `data-raw/r_parity.jl` — presumably a numeric cross-check script against the R `hcm`
  package; check its contents before assuming it's current if cross-validating the two
  implementations.
- `CausalGraphs.jl` is an unregistered dependency pulled via `[sources]` in `Project.toml`
  (URL-based), so `Pkg.instantiate()` needs network access to `github.com/xiangao/CausalGraphs.jl`
  the first time.
