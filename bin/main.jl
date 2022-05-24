@info "Julia version: " VERSION
@info "@__DIR__: " @__DIR__
@info "readdir(@__DIR__): " readdir(@__DIR__)

ENV["GENIE_ENV"] = "dev"
ENV["BASEPATH"] = "/proxy/8000/"
 
@info "Initializing packages"
using Revise
using ScoringEngineDemo
using BSON
using HTTP
using Sockets
using JSON3
using DataFrames
using Stipple
using StippleUI
using StipplePlotly
using PlotlyBase
using Random

using ShapML
using Weave

using StatsBase: sample
using Statistics: mean, std

include("setup.jl")
includet("app.jl")
includet("ui.jl")

# Stipple.Genie.config.base_path = "/proxy/8000/"
Genie.config.base_path = "/proxy/8000/"

route("/") do
    Model |> init |> handlers |> ui |> html
end

Genie.startup(8000, "0.0.0.0", async=false)
# down()