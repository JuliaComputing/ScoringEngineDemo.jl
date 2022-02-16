using Revise
using ScoringEngineDemo
using DataFrames
using Statistics
using StatsBase: sample
using BSON
using CairoMakie
using EvoTrees
using Random

global targetname = "event"

df_tot = ScoringEngineDemo.load_data("assets/training_data.csv")

# set target
transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

norm_feats = ["vh_age", "vh_value", "vh_speed", "vh_weight", "drv_age1",
    "pol_no_claims_discount", "pol_coverage", "density", 
    "drv_exp_yrs", "pol_duration", "pol_sit_duration",
    "drv_sex1", "has_drv2", "is_drv2_male"]

# train/eval split
Random.seed!(123)
df_train, df_eval = ScoringEngineDemo.data_splits(df_tot, 0.9)

preproc = ScoringEngineDemo.build_preproc(df_train, norm_feats = norm_feats)
adapter = ScoringEngineDemo.build_adapter_gbt(norm_feats, targetname)

df_train_pre = preproc(df_train)

density(collect(skipmissing(df_train_pre.vh_age)))
density(collect(skipmissing(df_train_pre.drv_age1)))

BSON.bson("assets/preproc-gbt.bson", Dict(:preproc => preproc))
BSON.bson("assets/adapter-gbt.bson", Dict(:adapter => adapter))