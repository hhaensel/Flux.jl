# Cost functions
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
    msle(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ)))

The loss corresponding to mean squared logarithmic errors, calculated as

    agg((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2)

The `ϵ` term provides numerical stability.
Penalizes an under-predicted estimate more than an over-predicted estimate.
"""
msle(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ))) = agg((log.(ŷ .+ ϵ) .- log.(y .+ ϵ)).^2)


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
   temp = abs_error .<  δ
   x = ofeltype(ŷ, 0.5)
   agg(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp))
end

wsum(w::Nothing, x; dims) = sum(x, dims=dims)
wsum(w::Number, x; dims) = w .* sum(x, dims=dims)
wsum(w::AbstractArray, x; dims) = sum( w .* x, dims=dims)

"""
    crossentropy(ŷ, y; weight=nothing, dims=1, ϵ=eps(eltype(ŷ)), agg=mean)

Return the cross entropy between the given probability distributions;
calculated as

    agg(.-sum(weight .* y .* log.(ŷ .+ ϵ); dims=dims))agg=mean, 

`weight` can be `nothing`, a number or an array.
`weight=nothing` acts like `weight=1` but is faster.

See also: [`Flux.logitcrossentropy`](@ref), [`Flux.binarycrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)
"""
function crossentropy(ŷ, y; dims=1, agg=mean, ϵ=eps(eltype(ŷ)), weight=nothing)
    agg(.-wsum(weight, y .* log.(ŷ .+ ϵ); dims=dims))
end

"""
    logitcrossentropy(ŷ, y; weight=nothing, agg=mean, dims=1)

Return the crossentropy computed after a [`Flux.logsoftmax`](@ref) operation;
calculated as

    agg(.-sum(weight .* y .* logsoftmax(ŷ; dims=dims); dims=dims))

`logitcrossentropy(ŷ, y)` is mathematically equivalent to
[`Flux.crossentropy(softmax(log.(ŷ)), y)`](@ref) but it is more numerically stable.

See also: [`Flux.crossentropy`](@ref), [`Flux.binarycrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)
"""
function logitcrossentropy(ŷ, y; dims=1, agg=mean, weight=nothing)
    agg(.-wsum(weight, y .* logsoftmax(ŷ; dims=dims); dims=dims))
end

"""
    binarycrossentropy(ŷ, y; ϵ=eps(ŷ))

Return ``-y*\\log(ŷ + ϵ) - (1-y)*\\log(1-ŷ + ϵ)``. The `ϵ` term provides numerical stability.

Typically, the prediction `ŷ` is given by the output of a [`sigmoid`](@ref) activation.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.logitbinarycrossentropy`](@ref)
"""
function binarycrossentropy(ŷ, y; agg=mean, ϵ=eps(eltype(ŷ)))
    agg(@.(-y*log(ŷ+ϵ) - (1-y)*log(1-ŷ+ϵ)))
end

# Re-definition to fix interaction with CuArrays.
# CuArrays.@cufunc binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)

"""
    logitbinarycrossentropy(ŷ, y; agg=mean)

`logitbinarycrossentropy(ŷ, y)` is mathematically equivalent to
[`Flux.binarycrossentropy(σ(log(ŷ)), y)`](@ref) but it is more numerically stable.

See also: [`Flux.crossentropy`](@ref), [`Flux.logitcrossentropy`](@ref), [`Flux.binarycrossentropy`](@ref)
"""
function logitbinarycrossentropy(ŷ, y; agg=mean)
    agg(@.((1-y)*ŷ - logsigmoid(ŷ)))
end
# Re-definition to fix interaction with CuArrays.
# CuArrays.@cufunc logitbinarycrossentropy(ŷ, y) = (1 - y)*ŷ - logσ(ŷ)


"""
    kldivergence(ŷ, y; dims=1, agg=mean, ϵ=eps(eltype(ŷ)))

Return the
[Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
between the given arrays interpreted as probability distributions.

KL divergence is a measure of how much one probability distribution is different
from the other.
It is always non-negative and zero only when both the distributions are equal
everywhere.
"""
function kldivergence(ŷ, y; dims=1, agg=mean, ϵ=eps(eltype(ŷ)))
  entropy = agg(sum(y .* log.(y .+ ϵ), dims=dims))
  cross_entropy = crossentropy(ŷ, y; dims=dims, agg=agg, ϵ=ϵ)
  return entropy + cross_entropy
end

"""
    poisson_loss(ŷ, y; agg=mean)

# Return how much the predicted distribution `ŷ` diverges from the expected Poisson
# distribution `y`; calculated as `sum(ŷ .- y .* log.(ŷ)) / size(y, 2)`.
REDO
[More information.](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/poisson).
"""
poisson_loss(ŷ, y; agg=mean) = agg(ŷ .- y .* log.(ŷ))

@deprecate poisson poisson_loss

"""
    hinge(ŷ, y; agg=mean)

Return the [hinge loss](https://en.wikipedia.org/wiki/Hinge_loss) given the
prediction `ŷ` and true labels `y` (containing 1 or -1); calculated as
`agg(max.(0, 1 .- ŷ .* y))`.

See also: [`squared_hinge`](@ref)
"""
hinge(ŷ, y; agg=mean) = agg(max.(0, 1 .-  ŷ .* y))

"""
    squared_hinge(ŷ, y; agg=mean)

Return the squared hinge loss given the prediction `ŷ` and true labels `y`
(containing 1 or -1); calculated as `agg((max.(0, 1 .- ŷ .* y)).^2))`.

See also: [`hinge`](@ref)
"""
squared_hinge(ŷ, y; agg=mean) = agg((max.(0, 1 .- ŷ .* y)).^2)

"""
    dice_coeff_loss(ŷ, y; smooth=1)

Return a loss based on the dice coefficient.
Used in the [V-Net](https://arxiv.org/pdf/1606.04797v1.pdf) image segmentation
architecture.
Similar to the F1_score. Calculated as:
    1 - 2*sum(|ŷ .* y| + smooth) / (sum(ŷ.^2) + sum(y.^2) + smooth)`
"""
dice_coeff_loss(ŷ, y; smooth=ofeltype(ŷ, 1.0)) = 1 - (2*sum(y .* ŷ) + smooth) / (sum(y.^2) + sum(ŷ.^2) + smooth) #TODO

"""
    tversky_loss(ŷ, y; β=0.7)

Return the [Tversky loss](https://arxiv.org/pdf/1706.05721.pdf).
Used with imbalanced data to give more weight to false negatives.
Larger β weigh recall higher than precision (by placing more emphasis on false negatives)
Calculated as:
    1 - sum(|y .* ŷ| + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1)
"""
tversky_loss(ŷ, y; β=ofeltype(ŷ, 0.7)) = 1 - (sum(y .* ŷ) + 1) / (sum(y .* ŷ + β*(1 .- y) .* ŷ + (1 - β)*y .* (1 .- ŷ)) + 1) #TODO