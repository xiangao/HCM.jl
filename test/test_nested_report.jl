@testset "nested report" begin
    draws = (λclass=randn(200).+0.4, λschool=randn(200).-0.2,
             σschool=abs.(randn(200)).+1, σclass=abs.(randn(200)).+0.5,
             β0=zeros(200), β1=fill(0.5,200), b_school=zeros(200,3), b_class=zeros(200,6), p_class=fill(0.5,200,6))
    s = nested_diagnostics(draws)
    @test all(k -> haskey(s,k), (:λclass,:λschool,:σschool,:σclass))
    @test abs(s.λclass.mean - mean(draws.λclass)) < 1e-9
    @test s.λclass.q025 < s.λclass.mean < s.λclass.q975
end
