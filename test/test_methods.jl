@testset "methods" begin
    draws = randn(1000) .* 0.1 .+ 0.5
    s = summarize_ate(draws)
    @test haskey(s, :mean) && haskey(s, :q025) && haskey(s, :q975)
    @test abs(s.mean - 0.5) < 0.05
    @test s.q025 < s.mean < s.q975
end
