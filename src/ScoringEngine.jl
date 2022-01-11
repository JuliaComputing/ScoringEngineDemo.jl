module ScoringEngine

using DataFrames
using Flux
using EvoTrees

export logit

include("preproc-utils.jl")
include("preproc.jl")
include("model.jl")
include("inference.jl")

end # module
