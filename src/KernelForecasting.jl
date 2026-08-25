module KernelForecasting

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

end
