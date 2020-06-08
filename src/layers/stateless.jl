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
# Possible deprecation path: `normalise(x; dims=1)` -> `normalise(x; dims)` -> `normalise(x; dims=size(x)[end])`  
"""
    normalise(x; dims)

Normalise `x` to mean 0 and standard deviation 1 across the dimensions given by `dims`.
Defaults to normalising over columns.

```jldoctest
julia> a = reshape(collect(1:9), 3, 3)
3×3 Array{Int64,2}:
 1  4  7
 2  5  8
 3  6  9

julia> Flux.normalise(a)
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