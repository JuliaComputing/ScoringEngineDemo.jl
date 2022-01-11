using Revise
using ScoringEngine
using DataFrames
using Statistics
using StatsBase: sample
using BSON
using CairoMakie
using Random

using Flux
using Flux: update!

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
preproc_adapt_flux = ScoringEngine.build_preproc_adapt_flux(norm_feats, targetname)

BSON.bson("assets/preproc.bson", Dict(:preproc => preproc))
BSON.bson("assets/preproc-adapt-flux.bson", Dict(:preproc_adapt => preproc_adapt_flux))

preproc(df_train)
preproc(df_eval)

x_train, y_train = preproc_adapt_flux(df_train, true)
x_eval, y_eval = preproc_adapt_flux(df_eval, true)

dtrain = Flux.Data.DataLoader((x_train, y_train), batchsize = 1024, shuffle = true)
deval = Flux.Data.DataLoader((x_eval, y_eval), batchsize = 1024, shuffle = false)

m = Chain(
    Dense(10, 128, relu),
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

opt = ADAM(5e-4)
θ = params(m)

for i in 1:25
    train_loop!(m, θ, opt, loss, dtrain = dtrain, deval = deval)
end

BSON.bson("assets/model-flux.bson", Dict(:model => m))


# preproc! = BSON.load("assets/preproc.bson", ScoringEngine)[:preproc!]
# preproc_adapt_flux = BSON.load("assets/preproc_adapt_flux.bson", ScoringEngine)[:preproc_adapt_flux]

# preproc!(df_eval)
# preproc_adapt_flux(df_eval)
# preproc_adapt_flux(df_eval, true)