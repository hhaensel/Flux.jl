"""
    mae(ŷ, y; agg=mean)

Return the loss corresponding to mean absolute error: 

    agg(abs.(ŷ .- y))
"""
mae(ŷ, y; agg=mean) = agg(abs.(ŷ .- y))

"""
    mse(ŷ, y; agg=mean)

Return the loss corresponding to mean square error: 
    
    agg((ŷ .- y).^2)
"""
mse(ŷ, y; agg=mean) = agg((ŷ .- y).^2)

"""
    msle(ŷ, y; agg=mean, ϵ=eps(ŷ))

The loss corresponding to mean squared logarithmic errors, calculated as

    agg((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2)

The `ϵ` term provides numerical stability.
Penalizes an under-estimation more than an over-estimatation.
"""
msle(ŷ, y; agg=mean, ϵ=epseltype(ŷ)) = agg((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2)

"""
    huber_loss(ŷ, y; δ=1, agg=mean)

Return the mean of the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
given the prediction `ŷ` and true values `y`.

                 | 0.5 * |ŷ - y|,            for |ŷ - y| <= δ
    Huber loss = |
                 |  δ * (|ŷ - y| - 0.5 * δ), otherwise
"""
function huber_loss(ŷ, y; agg=mean, δ=ofeltype(ŷ, 1))
   abs_error = abs.(ŷ .- y)
   #TODO: remove dropgrad when Zygote can handle this function with CuArrays
   temp = Zygote.dropgrad(abs_error .<  δ)
   x = ofeltype(ŷ, 0.5)
   agg(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp))
end

"""
    crossentropy(ŷ, y; dims=1, ϵ=eps(ŷ), agg=mean)

Return the cross entropy between the given probability distributions;
calculated as

    agg(-sum(y .* log.(ŷ .+ ϵ); dims=dims))

Cross entropy is tipically used as a loss in multi-class classification,
in which case the labels `y` are given in a one-hot format. 
`dims` specifies the dimension (or the dimensions) containing the class probabilities.
The prediction `ŷ` is supposed to sum to one across `dims`,
as would be the case with the output of a [`softmax`](@ref) operation. 

Use of [`logitcrossentropy`](@ref) is recomended over `crossentropy` for 
numerical stability.

See also: [`Flux.logitcrossentropy`](@ref), [`Flux.bce_loss`](@ref), [`Flux.logitbce_loss`](@ref)
"""
function crossentropy(ŷ, y; dims=1, agg=mean, ϵ=epseltype(ŷ))
    agg(.-sum(xlogy.(y, ŷ .+ ϵ); dims=dims))
end

"""
    logitcrossentropy(ŷ, y; dims=1, ϵ=eps(ŷ), agg=mean)

Return the crossentropy computed after a [`Flux.logsoftmax`](@ref) operation;
calculated as 

    agg(.-sum(y .* logsoftmax(ŷ; dims=dims); dims=dims))

`logitcrossentropy(ŷ, y)` is mathematically equivalent to
[`Flux.crossentropy(softmax(ŷ), y)`](@ref) but it is more numerically stable.

See also: [`Flux.crossentropy`](@ref), [`Flux.bce_loss`](@ref), [`Flux.logitbce_loss`](@ref)
"""
function logitcrossentropy(ŷ, y; dims=1, agg=mean)
    agg(.-sum(y .* logsoftmax(ŷ; dims=dims); dims=dims))
end

"""
    bce_loss(ŷ, y; agg=mean, ϵ=eps(ŷ))

Return the binary cross-entropy loss, computer as 

    agg(@.(-y*log(ŷ + ϵ) - (1-y)*log(1-ŷ + ϵ)))
    
The `ϵ` term provides numerical stability.

Typically, the prediction `ŷ` is given by the output of a [`sigmoid`](@ref) activation.

Use of `logitbce_loss` is recomended over `bce_loss` for numerical stability.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.logitbce_loss`](@ref)
"""
function bce_loss(ŷ, y; agg=mean, ϵ=epseltype(ŷ))
    agg(@.(-xlogy(y, ŷ+ϵ) - xlogy(1-y, 1-ŷ+ϵ)))
end
# Re-definition to fix interaction with CuArrays.
# CuArrays.@cufunc bce_loss(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

"""
    logitbce_loss(ŷ, y; agg=mean)

Mathematically equivalent to
[`Flux.bce_loss(σ(ŷ), y)`](@ref) but is more numerically stable.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.bce_loss`](@ref)
```
"""
function logitbce_loss(ŷ, y; agg=mean)
    agg(@.((1-y)*ŷ - logσ(ŷ)))
end
# Re-definition to fix interaction with CuArrays.
# CuArrays.@cufunc logitbce_loss(ŷ, y) = (1 - y)*ŷ - logσ(ŷ)


"""
    kldivergence(ŷ, y; agg=mean)

Return the
[Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
between the given probability distributions.

KL divergence is a measure of how much one probability distribution is different
from the other.
It is always non-negative and zero only when both the distributions are equal
everywhere.
"""
function kldivergence(ŷ, y; dims=1, agg=mean, ϵ=epseltype(ŷ))
  entropy = agg(sum(xlogx.(y), dims=dims))
  cross_entropy = crossentropy(ŷ, y; dims=dims, agg=agg, ϵ=ϵ)
  return entropy + cross_entropy
end

"""
    poisson_loss(ŷ, y)

# Return how much the predicted distribution `ŷ` diverges from the expected Poisson
# distribution `y`; calculated as `sum(ŷ .- y .* log.(ŷ)) / size(y, 2)`.
REDO
[More information.](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson_loss).
"""
poisson_loss(ŷ, y; agg=mean) = agg(ŷ .- xlogy.(y, ŷ))

"""
    hinge_loss(ŷ, y; agg=mean)

Return the [hinge_loss loss](https://en.wikipedia.org/wiki/Hinge_loss) given the
prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as
`sum(max.(0, 1 .- ŷ .* y)) / size(y, 2)`.

See also: [`squared_hinge_loss`](@ref)
"""
hinge_loss(ŷ, y; agg=mean) = agg(max.(0, 1 .-  ŷ .* y))

"""
    squared_hinge_loss(ŷ, y)

Return the squared hinge_loss loss given the prediction `ŷ` and true labels `y`
(containing 1 or -1); calculated as `sum((max.(0, 1 .- ŷ .* y)).^2) / size(y, 2)`.

See also: [`hinge_loss`](@ref)
"""
squared_hinge_loss(ŷ, y; agg=mean) = agg((max.(0, 1 .- ŷ .* y)).^2)

"""
    dice_coeff_loss(ŷ, y; smooth=1)

Return a loss based on the dice coefficient.
Used in the [V-Net](https://arxiv.org/pdf/1606.04797v1.pdf) image segmentation
architecture.
Similar to the F1_score. Calculated as:

    1 - 2*sum(|ŷ .* y| + smooth) / (sum(ŷ.^2) + sum(y.^2) + smooth)
"""
dice_coeff_loss(ŷ, y; smooth=ofeltype(ŷ, 1.0)) = 1 - (2*sum(y .* ŷ) + smooth) / (sum(y.^2) + sum(ŷ.^2) + smooth) #TODO agg

"""
    tversky_loss(ŷ, y; β=0.7)

Return the [Tversky loss](https://arxiv.org/pdf/1706.05721.pdf).
Used with imbalanced data to give more weight to false negatives.
Larger β weigh recall more than precision (by placing more emphasis on false negatives)
Calculated as:
    1 - sum(|y .* ŷ| + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)
"""
function tversky_loss(ŷ, y; β=ofeltype(ŷ, 0.7))
    #TODO add agg
    num = sum(y .* ŷ) + 1
    den = sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1
    1 - num / den
end
             
"""
    xlogx(x)

Return `x * log(x)` for `x ≥ 0`, handling `x = 0` by taking the downward limit.
"""
function xlogx(x)
  result = x * log(x)
  ifelse(iszero(x), zero(result), result)
end

CuArrays.@cufunc function xlogx(x)
  result = x * log(x)
  ifelse(iszero(x), zero(result), result)
end

"""
    xlogy(x, y)

Return `x * log(y)` for `y > 0` with correct limit at `x = 0`.
"""
function xlogy(x, y)
  result = x * log(y)
  ifelse(iszero(x), zero(result), result)
end

CuArrays.@cufunc function xlogy(x, y)
  result = x * log(y)
  ifelse(iszero(x), zero(result), result)
end

@adjoint function broadcasted(::typeof(xlogy), x::Zygote.Numeric, y::Zygote.Numeric)
  res = xlogy.(x, y)
  res, Δ -> (nothing, Zygote.unbroadcast(x, xlogy.(Δ, y)), Zygote.unbroadcast(y, Δ .* x ./ y))
end


"""
    flatten(x::AbstractArray)

Reshape arbitrarly-shaped input into a matrix-shaped output
preserving the last dimension size. 
Equivalent to `reshape(x, :, size(x)[end])`.
"""
function flatten(x::AbstractArray)
  return reshape(x, :, size(x)[end])
end

# TODO normalise over last dimension is typically what you want to do. 
# Deprecation path: `normalise(x; dims=1)` -> `normalise(x; dims)` -> `normalise(x; dims=size(x)[end])`  
"""
    normalise(x; dims, ϵ=1e-6)

Normalise `x` to mean 0 and standard deviation 1 across the dimensions given by `dims`.
Defaults to normalising over columns.
`ϵ` is a small additive factor added to the denominator for numerical stability.

```jldoctest
julia> a = reshape(collect(1:9), 3, 3)
3×3 Array{Int64,2}:
 1  4  7
 2  5  8
 3  6  9

julia> Flux.normalise(a, dims=1)
3×3 Array{Float64,2}:
 -1.22474  -1.22474  -1.22474
  0.0       0.0       0.0
  1.22474   1.22474   1.22474

julia> Flux.normalise(a, dims=2)
3×3 Array{Float64,2}:
 -1.22474  0.0  1.22474
 -1.22474  0.0  1.22474
 -1.22474  0.0  1.22474
```
"""
function normalise(x::AbstractArray; dims, ϵ=ofeltype(x, 1e-6))
  μ′ = mean(x, dims=dims)
    #   σ′ = std(x, dims=dims, mean=μ′, corrected=false) # use this when #478 gets merged
  σ′ = std(x, dims=dims, corrected=false)
  return (x .- μ′) ./ (σ′.+ ϵ)
end