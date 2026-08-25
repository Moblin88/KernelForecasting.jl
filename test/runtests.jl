using KernelForecasting
using Test
using LinearAlgebra
using Random

@testset "KernelForecasting.jl" begin

    Random.seed!(30)

    @testset "GaussianKernel" begin
        σ² = rand()
        k = GaussianKernel(σ²)
        x = rand(10, 4)
        y = rand(10, 4)
        @test k(x, y) ≈ exp(-sum(abs2.(x .- y))/(2*σ²))
    end

    @testset "LaplaceKernel" begin
        σ = rand()
        k = LaplaceKernel(σ)
        x = rand(10, 4)
        y = rand(10, 4)
        @test k(x, y) ≈ exp(-sum(abs.(x .- y))/(2*σ/sqrt(π)))
    end

    @testset "precompute_kernels" begin
        data = rand(10, 4)
        lags = 3
        n = size(data, 1)
        k = GaussianKernel(rand())
        pk = KernelForecasting.precompute_kernels(k, data, lags)
        @test length(pk) == lags^2 + (n-lags)*(lags+1)
        for (i,j) in zip(range(1; length = n - lags), range(1; length = n - lags, step = lags+1))
            kmatrix = [@views k(data[x, :], data[y, :]) for x in range(i; length = lags), y in range(i; length = lags)]
            kmatrixview = Symmetric(reshape(view(pk, range(j; length = lags^2)),lags,lags),:U)
            @test kmatrixview ≈ kmatrix
        end
        for (i,j) in zip(range(stop = n, length = n - lags),range(stop=length(pk)-lags, length = n - lags, step = lags + 1))
            kvector = [@views k(data[x, :], data[i, :]) for x in range(stop = i-1, length = lags)]
            kvectorview = view(pk, range(j; length = lags))
            @test kvectorview ≈ kvector
        end
    end

    @testset "LinearMatrixOperator" begin
        lag_embedding = rand(10, 4)
        edim = size(lag_embedding, 2)
        lags = 3
        n = size(lag_embedding, 1)
        kernels = rand(lags^2 + (n-lags)*(lags+1))
        op = KernelForecasting.LinearMatrixOperator(kernels, lag_embedding)
        model = rand(lags, edim)
        out = zeros(lags, edim)
        op(out, model)
        matoperator = zeros(eltype(out), lags*edim, lags*edim)
        
        for (i,j) in zip(range(1; length = n - lags, step = lags +1),range(1; length = n - lags))
            kmatrixview = Symmetric(reshape(view(op.kernels, range(i; length = lags^2)),lags,lags),:U)
            pview = view(op.lag_embedding, j, :)
            matoperator += kron(pview*pview', kmatrixview)
        end

        expected = matoperator * vec(model)
        @test vec(out) ≈ expected
    end
end
