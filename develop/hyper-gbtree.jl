@info "Initializing packages"
using ScoringEngineDemo
using BSON
using CSV
using DataFrames
using Random

using Distributed
@everywhere using EvoTrees

results_path = joinpath(@__DIR__, "results")
isdir(results_path) || mkdir(results_path)
ENV["RESULTS_FILE"] = results_path

@info "nworkers" nworkers()
@info "workers" workers()

@info "Initializing assets"
const assets_path = joinpath(@__DIR__, "..", "assets")
const preproc_gbt = BSON.load(joinpath(assets_path, "preproc-gbt.bson"), @__MODULE__)[:preproc]
const adapter_gbt = BSON.load(joinpath(assets_path, "adapter-gbt.bson"), @__MODULE__)[:adapter]

df_tot = ScoringEngineDemo.load_data(joinpath(assets_path, "training_data.csv"))

# set target
transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

# train/eval split
Random.seed!(123)
df_train, df_eval = ScoringEngineDemo.data_splits(df_tot, 0.9)

df_train = preproc_gbt(df_train)
df_eval = preproc_gbt(df_eval)

x_train, y_train = adapter_gbt(df_train, true)
x_eval, y_eval = adapter_gbt(df_eval, true)

nrounds = 3000

@everywhere function fit(; nrounds, max_depth, x_train, y_train, x_eval, y_eval)

    config = EvoTreeRegressor(
        loss = :logistic, metric = :logloss,
        nrounds = nrounds, nbins = 100,
        λ = 0.5, γ = 0.1, η = 0.05,
        max_depth = max_depth, min_weight = 1.0,
        rowsample = 0.5, colsample = 0.8)
    
    m = fit_evotree(config, x_train, y_train; X_eval = x_eval, Y_eval = y_eval, print_every_n = 25, early_stopping_rounds = 100)

    return (
        eval_metric = m.metric.metric,
        max_depth = max_depth,
        m = m)
end

[@spawnat p x_train = x_train for p in workers()]
[@spawnat p y_train = y_train for p in workers()]
[@spawnat p x_eval = x_eval for p in workers()]
[@spawnat p y_eval = y_eval for p in workers()]

max_depth_list = 2:9

@time results = pmap(max_depth_list) do max_depth
    fit(; nrounds, max_depth, x_train, y_train, x_eval, y_eval)
end

df_results = map(results) do n
    (max_depth = n[:max_depth], eval_metric = n[:eval_metric])
end |> DataFrame

m_best = results[findmin(df_results[:, :eval_metric])[2]][:m]

CSV.write(joinpath(results_path, "hyper-gbt.csv"), df_results)
BSON.bson(joinpath(results_path, "model-gbt.bson"), Dict(:model => m_best))