@deprecate param(x) x
@deprecate data(x) x
@deprecate poisson poisson_loss
@deprecate hinge hinge_loss
@deprecate squared_hinge squared_hinge_loss_loss
@deprecate binarycrossentropy(ŷ, y) bce_loss(ŷ, y, agg=identity)
@deprecate logitbinarycrossentropy(ŷ, y) logitbce_loss(ŷ, y, agg=identity)
