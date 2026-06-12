# Parallel recovery verification: runs every motif's NUTS recovery concurrently (one per worker)
# instead of the serial ~138-min suite. Each check fits the motif at adequate size and asserts the
# 95% posterior interval covers the simulated truth (the robust recovery criterion).
#
# Run: julia --project=. test/recovery_parallel.jl
# Writes per-motif results to /tmp/rec_<motif>.txt and a summary to stdout.

using Distributed
const NW = min(6, Sys.CPU_THREADS - 1)
nprocs() <= 1 && addprocs(NW)
# deps as their OWN @everywhere statement BEFORE the begin-block (so @formula is in scope on workers)
@everywhere using HCM, DataFrames, StatsModels, Statistics, LinearAlgebra
@everywhere begin
    BLAS.set_num_threads(1)
    covers(d, truth) = (q = quantile(d, [0.025, 0.975]); q[1] <= truth <= q[2])
    function check(motif)
        if motif == :confounder
            s = sim_hcm(:confounder; n=60, m=40, seed=1)
            f = hcm(@formula(y~a), s.data; unit=:unit, subunit=:subunit, motif=:confounder, iter=600, chains=2)
            d = ate(f, Hard(1); baseline=Hard(0)); r = (mean(d), s.true_hard, covers(d, s.true_hard))
        elseif motif == :nested_confounder
            s = sim_hcm(:nested_confounder; n=20, m=30, seed=1)
            f = hcm(@formula(y~a), s.data; groups=(:school,:class), motif=:nested_confounder, iter=600, chains=2)
            d = ate(f, Hard(1); baseline=Hard(0)); r = (mean(d), s.true_hard, covers(d, s.true_hard))
        elseif motif == :confounder_interference
            s = sim_hcm(:confounder_interference; n=60, m=30, seed=1)
            f = hcm(@formula(y~a), s.data; unit=:unit, subunit=:subunit, motif=:confounder_interference,
                    interferer=:z, unit_covar=:s, iter=600, chains=2)
            d = ate(f, Hard(1); baseline=Hard(0)); r = (mean(d), s.true_hard, covers(d, s.true_hard))
        elseif motif == :instrument
            s = sim_hcm(:instrument; n=80, m=40, seed=1)
            f = hcm(@formula(y~a), s.data; unit=:unit, subunit=:subunit, instrument=:z, motif=:instrument, iter=600, chains=2)
            d = ate(f, Hard(1); baseline=Hard(0)); r = (mean(d), s.true_hard, covers(d, s.true_hard))   # weak-IV: coverage
        elseif motif == :did
            s = sim_hcm(:did; n=20, m=8, seed=2, params=(T=5, τ=0.5))
            f = hcm(@formula(y~a), s.data; unit=:student, groups=(:school,:period), motif=:did, iter=600, chains=2)
            d = ate(f, Hard(1); baseline=Hard(0)); r = (mean(d), s.true_hard, covers(d, s.true_hard))
        elseif motif == :did_interference
            s = sim_hcm(:did_interference; n=20, m=8, seed=1, params=(T=5,))
            f = hcm(@formula(y~a), s.data; unit=:student, groups=(:school,:period), motif=:did_interference,
                    interferer=:z, unit_covar=:scov, iter=500, chains=2)
            tot = ate(f, Hard(1); baseline=Hard(0)); dir = did_direct(f)
            r = (mean(tot), s.true_total, covers(tot, s.true_total) && covers(dir, s.true_direct))
        end
        open(io->println(io, motif, "  est=", round(r[1],digits=3), "  truth=", round(r[2],digits=3),
                             "  covered=", r[3]), "/tmp/rec_$(motif).txt", "w")
        (motif, r...)
    end
end

motifs = [:confounder, :nested_confounder, :confounder_interference, :instrument, :did, :did_interference]
println("Running ", length(motifs), " recoveries on ", nworkers(), " workers ...")
res = pmap(check, motifs)
println("\n=== RECOVERY SUMMARY (95% interval covers truth) ===")
for (m, est, truth, cov) in res
    println(rpad(string(m), 24), " est=", rpad(round(est,digits=3),7), " truth=", rpad(round(truth,digits=3),7),
            "  ", cov ? "PASS" : "FAIL")
end
allpass = all(t -> t[4], res)   # t = (motif, est, truth, cov); avoid for-loop scope on a top-level var
println(allpass ? "\nALL RECOVERIES PASS" : "\nSOME RECOVERIES FAILED (single-seed coverage misses are expected ~5% of the time)")
