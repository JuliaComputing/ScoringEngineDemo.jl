using Revise
using ScoringEngine
using DataFrames
using Statistics
using StatsBase: sample
using BSON
using CairoMakie
using EvoTrees
using Random

global targetname = "event"

df_tot = ScoringEngine.load_data("assets/training_data.csv")

# set target
transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

# train/eval split
Random.seed!(123)
df_train, df_eval = ScoringEngine.data_splits(df_tot, 0.9)

norm_feats = ["vh_age", "vh_value", "vh_speed", "vh_weight", "drv_age1",
    "population", "town_surface_area", "pol_no_claims_discount", "density", "pol_coverage"]

preproc = ScoringEngine.build_preproc(df_train, norm_feats = norm_feats)
preproc_adapt_gbt = ScoringEngine.build_preproc_adapt_gbt(norm_feats, targetname)

BSON.bson("assets/preproc.bson", Dict(:preproc => preproc))
BSON.bson("assets/preproc-adapt-gbt.bson", Dict(:preproc_adapt => preproc_adapt_gbt))

preproc(df_train)
preproc(df_eval)

x_train, y_train = preproc_adapt_gbt(df_train, true)
x_eval, y_eval = preproc_adapt_gbt(df_eval, true)

config = EvoTreeRegressor(
    loss = :logistic, metric = :logloss,
    nrounds = 2000, nbins = 100,
    λ = 0.5, γ = 0.1, η = 0.05,
    max_depth = 6, min_weight = 1.0,
    rowsample = 0.5, colsample = 0.8)

m = fit_evotree(config, x_train, y_train, X_eval = x_eval, Y_eval = y_eval, print_every_n = 25, early_stopping_rounds=100)
predict(m, x_eval)

BSON.bson("assets/model-gbt.bson", Dict(:model => m))
