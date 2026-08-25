using KernelForecasting
using Test

@testset "KernelForecasting.jl" begin
    @testset "GaussianKernel" begin
        k = GaussianKernel(1.5)
        x = [1.0, 2.0]
        y = [2.0, 2.0]
        @test k(x, y) ≈ exp(-1/(2*1.5))
    end

    @testset "LaplaceKernel" begin
        k = LaplaceKernel(1.5)
        x = [1.0, 2.0]
        y = [2.0, 3.0]
        @test k(x, y) ≈ exp(-2/(2*1.5/sqrt(π)))
    end
end
