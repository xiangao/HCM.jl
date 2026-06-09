@testset "estimand" begin
    @test link_inv(0.5, :gaussian) == 0.5
    @test link_inv(0.0, :bernoulli) ≈ 0.5

    # 2 draws × 2 units
    β0 = [0.0 1.0; 0.0 1.0]
    β1 = [0.5 1.0; 0.5 1.0]
    p  = [0.3 0.7; 0.3 0.7]

    a_hard = ate(β0, β1, p, :gaussian, Hard(1); baseline=Hard(0))
    @test a_hard ≈ [mean([0.5,1.0]), mean([0.5,1.0])]

    a_soft = ate(β0, β1, p, :gaussian, Soft(1, 0.2))
    expected = 0.2 .* [mean([0.5*(1-0.3), 1.0*(1-0.7)]), mean([0.5*(1-0.3), 1.0*(1-0.7)])]
    @test a_soft ≈ expected

    β0b = reshape([0.1, -0.2], 1, 2); β1b = reshape([0.6, 0.4], 1, 2); pb = reshape([0.5, 0.5], 1, 2)
    as = ate(β0b, β1b, pb, :bernoulli, Soft(1, 0.1))
    edo(ε) = mean([(1-ε)*(0.5*logistic(0.1+0.6)+0.5*logistic(0.1)) + ε*logistic(0.1+0.6),
                   (1-ε)*(0.5*logistic(-0.2+0.4)+0.5*logistic(-0.2)) + ε*logistic(-0.2+0.4)])
    @test as ≈ [edo(0.1) - edo(0.0)]
end
