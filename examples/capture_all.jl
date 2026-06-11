# Capture REAL fitted outputs for the docs, writing each result to its own file as it finishes
# (so one slow fit never blocks the others). Run: julia --project=examples examples/capture_all.jl
using Distributed
nprocs() <= 1 && addprocs(5)
@everywhere using HCM, DataFrames, StatsModels, Statistics, LinearAlgebra
@everywhere begin
    BLAS.set_num_threads(1)
    ss(d) = string("(mean=", round(mean(d),digits=3), ", q025=", round(quantile(d,0.025),digits=3),
                   ", q975=", round(quantile(d,0.975),digits=3), ")")
    rnd(x) = round(x, digits=3)
    writeout(name, s) = open(io->println(io, s), "/tmp/cap_$name.txt", "w")
end

@everywhere function run_task(name)
    if name == "confounder"
        sim = sim_hcm(:confounder; n=60, m=40, seed=1)
        fit = hcm(@formula(y~a), sim.data; unit=:unit, subunit=:subunit, motif=:confounder)
        writeout(name, string("hard ATE  = ", ss(ate(fit,Hard(1);baseline=Hard(0))), "\n",
            "true_hard = ", rnd(sim.true_hard), "\n",
            "compare_fe = ", compare_fe(fit), "\n",
            "soft(1,0.1) = ", ss(ate(fit,Soft(1,0.1)))))
    elseif name == "nested"
        sim = sim_hcm(:nested_confounder; n=20, m=30, seed=1)
        fit = hcm(@formula(y~a), sim.data; groups=(:school,:class), motif=:nested_confounder)
        d = nested_diagnostics(fit.draws)
        writeout(name, string("hard ATE  = ", ss(ate(fit,Hard(1);baseline=Hard(0))), "\n",
            "true_hard = ", rnd(sim.true_hard), "\n",
            "λclass  = (mean=", rnd(d.λclass.mean), ", q025=", rnd(d.λclass.q025), ", q975=", rnd(d.λclass.q975), ")\n",
            "λschool = (mean=", rnd(d.λschool.mean), ", q025=", rnd(d.λschool.q025), ", q975=", rnd(d.λschool.q975), ")"))
    elseif name == "instrument"
        sim = sim_hcm(:instrument; n=80, m=40, seed=1)
        fit = hcm(@formula(y~a), sim.data; unit=:unit, subunit=:subunit, instrument=:z, motif=:instrument)
        writeout(name, string("hard ATE (θa) = ", ss(ate(fit,Hard(1);baseline=Hard(0))), "\n",
            "true_hard = ", rnd(sim.true_hard), "\n",
            "compare_fe = ", compare_fe(fit)))
    elseif name == "interference"
        sim = sim_hcm(:confounder_interference; n=60, m=30, seed=1)
        fit = hcm(@formula(y~a), sim.data; unit=:unit, subunit=:subunit,
                  motif=:confounder_interference, interferer=:z, unit_covar=:s, iter=600, chains=2)
        writeout(name, string("interference-HCM hard ATE (total) = ", ss(ate(fit,Hard(1);baseline=Hard(0))), "\n",
            "true_hard (total) = ", rnd(sim.true_hard)))
    elseif name == "sutva"
        sim = sim_hcm(:confounder_interference; n=60, m=30, seed=1)
        fit = hcm(@formula(y~a), sim.data; unit=:unit, subunit=:subunit, motif=:confounder, iter=600, chains=2)
        writeout(name, string("confounder-HCM (SUTVA) hard ATE (direct only) = ", ss(ate(fit,Hard(1);baseline=Hard(0)))))
    end
    name
end

names = ["confounder", "nested", "instrument", "interference", "sutva"]
pmap(run_task, names)
println("ALL DONE")
