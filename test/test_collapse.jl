using HCM, Test
import CausalGraphs: ADMG

# helpers: treat edges as unordered/ordered sets for comparison
diset(g::ADMG) = Set(g.directed_edges)
biset(g::ADMG) = Set(Set{Symbol}([a, b]) for (a, b) in g.bidirected_edges)

@testset "collapse" begin
    g = hcm_graph(vertices=[:U, :A, :Y], subunit=[:A, :Y], hidden=[:U],
                  di_edges=[(:U, :A), (:U, :Y), (:A, :Y)])
    m = collapse(g)
    @test Set(m.vertices) == Set([:QA, :QY, :U])
    @test !m.observed[:U] && m.observed[:QA] && m.observed[:QY]
    # subunit→subunit edge A→Y is absorbed into the conditional label, not a Q→Q edge
    @test Set(m.directed_edges) == Set([(:U, :QA), (:U, :QY)])
    @test m.labels[:QY] == "Q^{y|a}"

    # interference: A → unit channel Z → Y  (paper Eq. 28 collapsed form)
    gi = hcm_graph(vertices=[:U, :A, :Z, :Y], subunit=[:A, :Y], hidden=[:U],
                   di_edges=[(:U, :A), (:U, :Y), (:A, :Y), (:A, :Z), (:Z, :Y)])
    mi = collapse(gi)
    @test Set(mi.directed_edges) == Set([(:U, :QA), (:U, :QY), (:QA, :Z), (:Z, :QY)])

    # instrument: subunit Z → A, unit outcome Y  (paper Eq. 33 collapsed form)
    gv = hcm_graph(vertices=[:U, :Z, :A, :Y], subunit=[:Z, :A], hidden=[:U],
                   di_edges=[(:Z, :A), (:U, :A), (:U, :Y), (:A, :Y)])
    mv = collapse(gv)
    @test m.observed[:QA]
    @test Set(mv.directed_edges) == Set([(:U, :QA), (:QZ, :Y), (:QA, :Y), (:U, :Y)])
    @test mv.labels[:QA] == "Q^{a|z}"   # A's subunit parent Z → conditional
end

@testset "latent_project" begin
    # fork through a hidden node ⇒ bidirected edge
    m1 = collapsed_model(vertices=[:U, :A, :B], hidden=[:U], di_edges=[(:U, :A), (:U, :B)])
    g1 = latent_project(m1)
    @test Set(g1.vertices) == Set([:A, :B])
    @test isempty(g1.directed_edges)
    @test biset(g1) == Set([Set([:A, :B])])

    # directed path through a hidden node ⇒ directed edge
    m2 = collapsed_model(vertices=[:A, :H, :B], hidden=[:H], di_edges=[(:A, :H), (:H, :B)])
    g2 = latent_project(m2)
    @test diset(g2) == Set([(:A, :B)])
    @test isempty(g2.bidirected_edges)

    # collider at a hidden node ⇒ NO edge between its parents
    m3 = collapsed_model(vertices=[:A, :C, :B], hidden=[:C], di_edges=[(:A, :C), (:B, :C)])
    g3 = latent_project(m3)
    @test isempty(g3.directed_edges) && isempty(g3.bidirected_edges)
end

@testset "identification verdicts (paper §4 + CAIRE)" begin
    c = hcm_confounder()
    rc = identify_hcm(c.graph; treatment=c.treatment, outcome=c.outcome, augments=c.augments)
    @test rc.verdict == :a_fixable           # backdoor on Q^{y|a}, Eq. 25
    @test rc.motif == :confounder

    i = hcm_interference()
    ri = identify_hcm(i.graph; treatment=i.treatment, outcome=i.outcome, augments=i.augments)
    @test ri.verdict == :p_fixable           # front-door, Eq. 30
    @test ri.motif == :confounder_interference

    v = hcm_instrument()
    rv = identify_hcm(v.graph; treatment=v.treatment, outcome=v.outcome,
                      augments=v.augments, marginalizes=v.marginalizes)
    @test rv.verdict == :a_fixable           # backdoor on Q^{a|z}, Eq. 34
    @test rv.motif == :instrument

    # CAIRE: the selection-law constraint (r observed) is what buys identification
    rcaire = identify_hcm(caire_model(r_observed=true); treatment=:qa, outcome=:y)
    @test rcaire.verdict == :a_fixable       # backdoor on reconstructed fitness r (Thm 1)
    @test Set(rcaire.admg.bidirected_edges) |> g -> Set(Set{Symbol}([a, b]) for (a, b) in g) ==
          Set([Set([:r, :y])])

    # drop the constraint (r hidden) ⇒ the IV graph, not nonparametrically identified
    rno = identify_hcm(caire_model(r_observed=false); treatment=:qa, outcome=:y)
    @test rno.verdict == :not_identified
end

@testset "mermaid rendering" begin
    g = hcm_confounder().graph
    s = to_mermaid(g)
    @test occursin("flowchart", s)
    @test occursin("subunit_plate", s)
    @test occursin("hidden", s)            # U rendered as hidden
    m = collapse(g)
    sm = to_mermaid(m)
    @test occursin("Q^{y|a}", sm)
end
