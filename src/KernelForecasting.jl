module KernelForecasting

using LinearAlgebra
export GaussianKernel, LaplaceKernel

struct GaussianKernel{T<:Real}
    σ²::T
end

function (k::GaussianKernel)(x, y)
    s = zero(real(promote_type(eltype(x), eltype(y))))
    for (u,v) in zip(x, y)
        s += abs2(u - v)
    end
    return exp(-s/(2*k.σ²))
end

struct LaplaceKernel{T<:Real}
    σ::T
end

function (k::LaplaceKernel)(x, y)
    s = zero(real(promote_type(eltype(x), eltype(y))))
    for (u,v) in zip(x, y)
        s += abs(u - v)
    end
    return exp(-s/(2*k.σ/sqrt(π)))
end

function precompute_kernels(k, data, lags)
    Base.require_one_based_indexing(data)
    n = size(data, 1)
    1<=lags <= n || throw(ArgumentError("lags must be between 1 and the number of rows in data"))
    mem1 = @views k(data[1, :], data[1, :])
    mem = Memory{typeof(mem1)}(undef, lags*(n-1) + n)
    # fill the first matrix
    mem[1] = mem1
    for j in 2:lags, i in 1:j
        mem[(j-1) * lags + i] = @views k(data[i, :], data[j, :])
    end
    #fill the remaining columns
    for j in lags+1:n, i in j-lags:j
        mem[(j-1) * lags + i] = @views k(data[i, :], data[j, :])
    end
    return mem
end

end
