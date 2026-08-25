using KernelForecasting
using Test
using LinearAlgebra

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

    @testset "precompute_kernels" begin
        data = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0]
        lags = 3
        n = size(data, 1)
        k = GaussianKernel(1.5)
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
        data = [1.0 0.0; 0.0 1.0; 1.0 1.0; 2.0 0.0]
        lags = 2
        n = size(data, 1)
        k = GaussianKernel(1.0)
        kernels = KernelForecasting.precompute_kernels(k, data, lags)

        # lag_embedding rows are a feature map for a real kernel matrix over
        # distinct lag positions, so lag_embedding*lag_embedding' ≈ that
        # kernel matrix (unit diagonal, since k(x,x) == 1, and unique rows,
        # since the underlying positions are distinct).
        lag_kernel = LaplaceKernel(1.0)
        lag_positions = [[0.0], [1.0], [2.0], [3.0]]
        lag_gram = [lag_kernel(x, y) for x in lag_positions, y in lag_positions]
        lag_embedding = Matrix(cholesky(Symmetric(lag_gram)).L)

        op = KernelForecasting.LinearMatrixOperator(kernels, lag_embedding)
        edim = size(lag_embedding, 2)
        model = reshape(1.0:(lags*edim), lags, edim)
        out = zeros(lags, edim)

        op(out, model)

        expected = zeros(size(out))
        for j in 1:(n - lags)
            kmatrix = [@views k(data[x, :], data[y, :]) for x in range(j; length = lags), y in range(j; length = lags)]
            p = lag_embedding[j, :]
            expected .+= kmatrix * model * p * p'
        end
        @test out ≈ expected
    end
end
