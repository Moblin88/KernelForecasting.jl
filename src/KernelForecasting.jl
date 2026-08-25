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
    mem = Vector{typeof(mem1)}(undef, lags*(n-1) + n)
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

struct LinearMatrixOperator{V<:AbstractVector{<:Real}, M<:AbstractMatrix{<:Real}, B<:Real}
    kernels::V
    lag_embedding::M
    buffer::Vector{B}
    model_buffer::Vector{B}
    function LinearMatrixOperator(kernels::V, lag_embedding::M) where {V<:AbstractVector{<:Real}, M<:AbstractMatrix{<:Real}}
        Base.require_one_based_indexing(kernels, lag_embedding)
        n = size(lag_embedding, 1)
        lags, remainder = divrem(length(kernels) - n, n-1)
        iszero(remainder) || throw(ArgumentError("length of kernels must be (n-1)*(lags+n) where n is the number of rows in lag_embedding"))
        buff_type = promote_type(eltype(kernels), eltype(lag_embedding))
        buffer = Vector{buff_type}(undef, lags)
        model_buffer = Vector{buff_type}(undef, lags)
        new{V,M,buff_type}(kernels, lag_embedding, buffer, model_buffer)
    end
end

function (op::LinearMatrixOperator)(out::AbstractMatrix, model::AbstractMatrix)
    Base.require_one_based_indexing(out, model)
    lags = size(op.buffer, 1)
    n = size(op.lag_embedding, 1)
    edim = size(op.lag_embedding, 2)
    size(out) == size(model) == (lags, edim) || throw(DimensionMismatch(lazy"out and model must be $lags x $edim matrices"))
    buff = op.buffer
    model_buff = op.model_buffer
    for (i,j) in zip(range(1; length = n - lags, step = lags + 1), range(1; length = n - lags))
        kmatrixview = Symmetric(reshape(view(op.kernels, range(i; length = lags^2)),lags,lags),:U)
        pview = view(op.lag_embedding, j, :)
        mul!(model_buff, model, pview)
        mul!(buff, kmatrixview, model_buff)
        mul!(out, buff, pview',one(promote_type(eltype(buff), eltype(pview'))), one(eltype(out)))
    end
    return out
end

end # module KernelForecasting
