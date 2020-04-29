## Loss Functions
Flux provides a large number of common loss functions used for training machine learning models. 

Most loss functions in Flux have an optional argument `agg`, denoting the type of aggregation performed over the
batch: 
```julia
loss(ŷ, y; agg=mean)
```

```@docs
Flux.mae
Flux.mse
Flux.msle
Flux.huber_loss
Flux.crossentropy
Flux.logitcrossentropy
Flux.binarycrossentropy
Flux.logitbinarycrossentropy
Flux.kldivergence
Flux.poisson_loss
Flux.hinge
Flux.squared_hinge
Flux.dice_coeff_loss
Flux.tversky_loss
```