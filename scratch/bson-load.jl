# using Revise
using ScoringEngineDemo
using DataFrames
using Statistics
using StatsBase: sample
using BSON
using EvoTrees


normA = ScoringEngineDemo.Normalizer(0.5, 0.1)
BSON.bson("assets/normA.bson", Dict(:norm => normA))

x1 = rand(5)
normA(x1)

normB = BSON.load("assets/normA.bson")[:norm]
normB(x1)

ScoringEngineDemo.normA(x1)
ScoringEngineDemo.preproc_flux
ScoringEngineDemo.infer_flux

const assets_path = joinpath(pkgdir(ScoringEngineDemo), "assets")
df_tot = begin
    df_tot = ScoringEngineDemo.load_data(joinpath(assets_path, "training_data.csv"))
    transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")
    dropmissing!(df_tot)
end
ScoringEngineDemo.infer_flux(df_tot)
ScoringEngineDemo.infer_gbt(df_tot)
