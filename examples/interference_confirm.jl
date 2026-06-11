# Confirm the package's estimators achieve the estimands plotted in interference_showcase.jl,
# at adequate sample size, for one representative interference strength (γ1 = -0.8).
#
#   interference-HCM  should ≈ τ_total(-0.8)  = 0.3 + (0.2+0.6)(-0.8) = -0.34   (total effect)
#   confounder-HCM    should ≈ τ_direct(-0.8) = 0.3 + 0.6(0.5)(-0.8)  =  0.06   (direct only)
#
# Run:  julia --project=examples examples/interference_confirm.jl   (minutes; NUTS)

using HCM, DataFrames, StatsModels, Statistics
ss(d) = (mean=round(mean(d),digits=3), q025=round(quantile(d,0.025),digits=3),
         q975=round(quantile(d,0.975),digits=3))

sim = sim_hcm(:confounder_interference; n=150, m=100, seed=20, params=(γ1=-0.8,))
open("/tmp/interference_confirm.txt", "w") do io
    println(io, "true_total  = ", round(sim.true_hard, digits=3)); flush(io)
    fi = hcm(@formula(y~a), sim.data; unit=:unit, subunit=:subunit,
             motif=:confounder_interference, interferer=:z, unit_covar=:s, iter=600, chains=2)
    println(io, "interference_HCM = ", ss(ate(fi, Hard(1); baseline=Hard(0)))); flush(io)
    fc = hcm(@formula(y~a), sim.data; unit=:unit, subunit=:subunit, motif=:confounder,
             iter=600, chains=2)
    println(io, "confounder_HCM   = ", ss(ate(fc, Hard(1); baseline=Hard(0)))); flush(io)
    println(io, "DONE"); flush(io)
end
