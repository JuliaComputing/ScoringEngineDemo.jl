module ScoringEngineDemo

using DataFrames
using Flux
using EvoTrees

using StatsBase: sample, quantile
using Statistics: mean, std
using PlotlyBase

export logit
export one_way_data, one_way_plot, one_way_plot_weights

const j_blue = "#4063D8"
const j_green = "#389826"
const j_purple = "#9558B2"
const j_red = "#CB3C33"

include("preproc-utils.jl")
include("preproc.jl")
include("model.jl")
include("inference.jl")
include("plots.jl")

end # module
