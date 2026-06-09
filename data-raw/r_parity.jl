# Optional faithful-port check: compare HCM.jl to the R `hcm` package on a matched DGP.
# Both should recover the same ground truth (Stan-NUTS vs Turing-NUTS won't match bit-for-bit).
# Run manually; requires the R package at ~/projects/software/hcm and Rscript.
using HCM, Statistics, StatsModels
sim = sim_hcm(:confounder; n=50, m=40, params=(β1=0.5,), seed=11)
fit = hcm(@formula(y ~ a), sim.data; unit=:unit, subunit=:subunit, iter=600, chains=2)
println("HCM.jl hard ATE = ", mean(ate(fit, Hard(1); baseline=Hard(0))), " ; truth = ", sim.true_hard)
