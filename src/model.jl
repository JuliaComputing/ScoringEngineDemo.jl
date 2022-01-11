"""
logit
"""
logit(x::Real) = 1 / (1 + exp(-x))
logit(x::AbstractVector) = 1 ./ (1 .+ exp.(-x))
