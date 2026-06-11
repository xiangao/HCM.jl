# Interference vs. SUTVA: what modelling the spillover channel buys you.
#
# A unit-level channel z carries treatment between subunits (the CONFOUNDER & INTERFERENCE
# motif): each subunit's treatment a_ij lifts the unit treatment rate ā_i, which drives z_i,
# which feeds back onto *every* subunit's outcome. The policy-relevant estimand is the TOTAL
# effect of do(a=1) vs do(a=0) — direct + spillover.
#
# The advantage of modelling interference is an ESTIMAND fact, exact for this DGP — no MCMC
# needed to make the point (and small/fast NUTS fits would only add noise). For
# y_ij = μ0 + u_i + δ0 z_i + (μ1 + δ1 z_i) a_ij,  z_i = γ0 + γ1 ā_i + γ2 s_i:
#
#   interference-HCM target (TOTAL effect, full-do through the channel):
#       τ_total(γ1)  = μ1 + (δ0 + δ1)·γ1
#   SUTVA confounder-HCM target (within-unit DIRECT effect, channel held fixed):
#       τ_direct(γ1) = μ1 + δ1·E[z_obs] = μ1 + δ1·(γ1·p_a)         (p_a = 0.5)
#
# They coincide at γ1 = 0 and fan apart as interference grows; the gap (δ0 + p_a·δ1)·γ1 is the
# spillover the SUTVA analysis throws away. We also show the naive aggregate OLS of unit-mean y
# on unit-mean a (a third line) computed as a large-sample plim.
#
# That the package's *estimators* achieve these targets at adequate sample size is verified by
# the interference recovery test in the suite; a single confirming fit is in
# examples/interference_confirm.jl.
#
# Run:  julia --project=examples examples/interference_showcase.jl   (seconds; no MCMC)

using HCM, DataFrames, Statistics, CairoMakie
CairoMakie.activate!(type="png")

# DGP defaults from sim_hcm(:confounder_interference) — keep in sync with src/sim.jl
const μ1, δ0, δ1, pa = 0.3, 0.2, 0.6, 0.5

τ_total(γ1)  = μ1 + (δ0 + δ1) * γ1          # interference-HCM estimand
τ_direct(γ1) = μ1 + δ1 * (γ1 * pa)          # SUTVA confounder-HCM estimand

# naive aggregate OLS plim: regress unit-mean y on unit-mean a in one large sample
function naive_plim(γ1; n=4000, m=200, seed=1)
    d = sim_hcm(:confounder_interference; n=n, m=m, seed=seed, params=(γ1=γ1,)).data
    g = combine(groupby(d, :unit), :y => mean => :yb, :a => mean => :ab)
    x = g.ab .- mean(g.ab); y = g.yb .- mean(g.yb)
    sum(x .* y) / sum(x .^ 2)
end

γgrid = range(0.0, -1.2; length=25)
tot = τ_total.(γgrid)
dir = τ_direct.(γgrid)
# naive aggregate OLS plim — reported in text, NOT plotted: in this randomized-treatment DGP
# (a ⊥ u) the aggregate ȳ–ā slope happens to absorb the channel path and tracks τ_total, so a
# line on the figure would misleadingly suggest it is a robust method. We print it for honesty.
γnaive = [0.0, -0.6, -1.2]
nai = naive_plim.(γnaive)
println("\nnaive aggregate OLS plim (NOT a robust method; tracks total here only because a⊥u):")
for (g, v) in zip(γnaive, nai); println("  γ1=", g, " → ", round(v, digits=3),
    "   (τ_total=", round(τ_total(g), digits=3), ", τ_direct=", round(τ_direct(g), digits=3), ")"); end

# print a results table (subset of γ1) for the vignette
println("\n## Estimands vs interference strength\n")
println("| γ₁ | interference-HCM (total) | SUTVA confounder-HCM (direct) | spillover gap |")
println("|---|---|---|---|")
for γ1 in [0.0, -0.3, -0.6, -0.9, -1.2]
    t, d = τ_total(γ1), τ_direct(γ1)
    println("| ", γ1, " | ", round(t, digits=3), " | ", round(d, digits=3),
            " | ", round(t - d, digits=3), " |")
end

# ── figure ────────────────────────────────────────────────────────────────────
fig = Figure(size=(780, 500), fontsize=15)
ax = Axis(fig[1, 1],
    xlabel="interference strength  γ₁   (channel  ā → z,  more negative = stronger spillover)",
    ylabel="average treatment effect of do(a=1) vs do(a=0)",
    title="Modelling interference recovers the total effect;\nassuming no spillover (SUTVA) recovers only the direct effect")
hlines!(ax, [0.0], color=(:gray, 0.5), linestyle=:dot)
lines!(ax, γgrid, tot, color=:dodgerblue, linewidth=3,
       label="interference-HCM  →  total effect  (correct)")
lines!(ax, γgrid, dir, color=:darkorange, linewidth=3,
       label="confounder-HCM (SUTVA)  →  direct effect only")
band!(ax, γgrid, dir, tot, color=(:red, 0.08))
text!(ax, -0.62, (τ_total(-0.62)+τ_direct(-0.62))/2; text="spillover\nthrown away by SUTVA",
      align=(:center, :center), fontsize=12, color=:firebrick)
axislegend(ax, position=:lb, framevisible=true)

assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)
out = joinpath(assets, "interference_vs_sutva.png")
save(out, fig)
println("\nFigure written to ", normpath(out))
