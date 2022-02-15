using Revise
using ScoringEngineDemo
using DataFrames
using Statistics
using StatsBase: sample
using BSON
using CairoMakie
using Random
using CUDA

using Flux
using Flux: update!
using ParameterSchedulers

global targetname = "event"

df_tot = ScoringEngineDemo.load_data("assets/training_data.csv")

# set target
transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

# train/eval split
Random.seed!(123)
df_train, df_eval = ScoringEngineDemo.data_splits(df_tot, 0.9)

norm_feats = ["vh_age", "vh_value", "vh_speed", "vh_weight", "drv_age1",
    "pol_no_claims_discount", "pol_coverage", "density", 
    # "drv_exp_yrs", 
    "pol_duration", "pol_sit_duration",
    "drv_sex1", "has_drv2", "is_drv2_male"]

preproc = ScoringEngineDemo.build_preproc(df_train, norm_feats = norm_feats)
adapter_flux = ScoringEngineDemo.build_adapter_flux(norm_feats, targetname)

# BSON.bson("assets/preproc-flux.bson", Dict(:preproc => preproc))
# BSON.bson("assets/preproc-adapt-flux.bson", Dict(:preproc_adapt => preproc_adapt_flux))

df_train = preproc(df_train)
df_eval = preproc(df_eval)

x_train, y_train = adapter_flux(df_train, true)
x_eval, y_eval = adapter_flux(df_eval, true)

dtrain = Flux.Data.DataLoader((x_train |> gpu, y_train |> gpu), batchsize = 1024, shuffle = true)
deval = Flux.Data.DataLoader((x_eval |> gpu, y_eval |> gpu), batchsize = 1024, shuffle = false)

m = Chain(
    Dense(size(x_train, 1), 128, relu),
    Dropout(0.5),
    Dense(128, 32, relu),
    SkipConnection(Dense(32, 32, relu), +),
    Dense(32, 1),
    x -> reshape(x, :))

function loss(m, x, y)
    l = mean(exp.(m(x)) .- m(x) .* y)
    return l
end

# cb() = @show(loss(X_eval, y_eval))
function logloss(data, m)
    logloss = 0.0
    count = 0
    for (x, y) in data
        logloss += sum(exp.(m(x)) .- m(x) .* y)
        count += size(x)[end]
    end
    return logloss / count
end

function train_loop!(m, θ, opt, loss; dtrain, deval = nothing)
    for d in dtrain
        grads = gradient(θ) do
            loss(m, d...)
        end
        update!(opt, θ, grads)
    end
    metric = deval === nothing ? logloss(deval, m) : logloss(deval, m)
    println(metric)
end

m = m |> gpu
# m, opt = BSON.load("assets/flux-test-gpu.bson")[:model], BSON.load("assets/flux-test-gpu.bson")[:opt]
opt = ADAM(5e-4)
θ = params(m)

for i in 1:8
    train_loop!(m, θ, opt, loss, dtrain = dtrain, deval = deval)
end

# BSON.bson("assets/model-flux.bson", Dict(:model => m))
# BSON.bson("assets/flux-test-gpu.bson", Dict(:model => m, :opt => opt))
# BSON.bson("assets/flux-test-cpu.bson", Dict(:model => m |> cpu, :opt => opt |> cpu))
