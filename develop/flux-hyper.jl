@info "Initializing packages"
using ScoringEngineDemo
using BSON
using JSON3
using CSV
using DataFrames
using Random

using Distributed
addprocs(8)
nworkers()
workers()

@everywhere using Statistics: mean
@everywhere using Flux
@everywhere using Flux: update!

@info "Initializing assets"
const assets_path = joinpath(@__DIR__, "..", "assets")
const preproc_flux = BSON.load(joinpath(assets_path, "preproc-flux.bson"), @__MODULE__)[:preproc]
const preproc_gbt = BSON.load(joinpath(assets_path, "preproc-gbt.bson"), @__MODULE__)[:preproc]

const preproc_adapt_flux = BSON.load(joinpath(assets_path, "preproc-adapt-flux.bson"), @__MODULE__)[:preproc_adapt]
const preproc_adapt_gbt = BSON.load(joinpath(assets_path, "preproc-adapt-gbt.bson"), @__MODULE__)[:preproc_adapt]

df_tot = ScoringEngineDemo.load_data(joinpath(assets_path, "training_data.csv"))

# set target
transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

# train/eval split
Random.seed!(123)
df_train, df_eval = ScoringEngineDemo.data_splits(df_tot, 0.9)

df_train = preproc_flux(df_train)
df_eval = preproc_flux(df_eval)

x_train, y_train = preproc_adapt_flux(df_train, true)
x_eval, y_eval = preproc_adapt_flux(df_eval, true)

dtrain = Flux.Data.DataLoader((x_train, y_train), batchsize = 1024, shuffle = true)
deval = Flux.Data.DataLoader((x_eval, y_eval), batchsize = 1024, shuffle = false)

@everywhere function loss(m, x, y)
    l = mean(exp.(m(x)) .- m(x) .* y)
    return l
end

# cb() = @show(loss(X_eval, y_eval))
@everywhere function logloss(data, m)
    logloss = 0.0
    count = 0
    for (x, y) in data
        logloss += sum(exp.(m(x)) .- m(x) .* y)
        count += size(x)[end]
    end
    return logloss / count
end

@everywhere function train_loop!(m, θ, opt, loss; dtrain, deval = nothing)
    for d in dtrain
        grads = gradient(θ) do
            loss(m, d...)
        end
        update!(opt, θ, grads)
    end
    metric = deval === nothing ? logloss(dtrain, m) : logloss(deval, m)
    println(metric)
end

@everywhere function fit(num_feats, h1, dtrain, deval)

    m = Chain(
        Dense(num_feats, h1, relu),
        Dropout(0.5),
        Dense(h1, 32, relu),
        SkipConnection(Dense(32, 32, relu), +),
        Dense(32, 1),
        x -> reshape(x, :))

    opt = ADAM(5e-4)
    θ = params(m)

    for i in 1:25
        train_loop!(m, θ, opt, loss, dtrain = dtrain, deval = deval)
    end

    eval_metric = logloss(deval, m)
    return eval_metric
end

num_feats = size(x_train, 1)
@time fit(num_feats, 128, dtrain, deval)

# @spawnat 2 dtrain = dtrain
# [@spawnat p dtrain = dtrain for p in workers()]
# [@spawnat p deval = deval for p in workers()]

[@spawnat p x_train = x_train for p in workers()]
[@spawnat p y_train = y_train for p in workers()]
[@spawnat p x_eval = x_eval for p in workers()]
[@spawnat p y_eval = y_eval for p in workers()]

h1_list = 32:32:256
length(h1_list)
@time results = pmap(h1_list) do h1
    dtrain = Flux.Data.DataLoader((x_train, y_train), batchsize = 1024, shuffle = true)
    deval = Flux.Data.DataLoader((x_eval, y_eval), batchsize = 1024, shuffle = false)
    fit(num_feats, h1, dtrain, deval)
end

df_results = DataFrame("eval_metric" => results, "h1" => h1_list)
CSV.write("hyper-flux.csv", df_results)
ENV["RESULTS_FILE"] = "hyper-flux.csv"