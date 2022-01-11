using CSV
using DataFrames
using Statistics: mean, std
using StatsBase: sample
using BSON

# const targetname = "event"

"""
Preproc
Preproc functor. Holds a vector of transform operations
"""
struct Preproc
    layers::Vector
end

function (p::Preproc)(df::DataFrame, ids = nothing)
    df = copy(df)
    ids = isnothing(ids) ? range(1, length(p.layers), step = 1) : ids
    if length(ids) == 1
        transform!(df, p.layers[ids[1]])
    else
        for layer in p.layers[ids]
            transform!(df, layer)
        end
    end
    return df
end

function build_preproc_adapt_flux(featnames, targetname)
    f = function f(df::DataFrame, include_target::Bool = false)
        data = collect(Array{Float32}(df[:, featnames])')
        if include_target
            target = df[:, targetname]
            return (data = data, target = target)
        else
            return data
        end
    end
    return f
end

function build_preproc_adapt_gbt(featnames, targetname)
    f = function f(df::DataFrame, include_target::Bool = false)
        data = collect(Array{Float32}(df[:, featnames]))
        if include_target
            target = df[:, targetname]
            return (data = data, target = target)
        else
            return data
        end
    end
    return f
end

"""
density(pop, area)
"""
density(pop, area) = pop ./ area

"""
Normalizer

Constructor store normalisation parameters based on input vector mean and std. 
Functor apply normalisation parameters to input vector
"""
struct Normalizer <: Function
    μ
    σ
end

Normalizer(x::AbstractVector) = Normalizer(mean(skipmissing(x)), std(skipmissing(x)))

function (m::Normalizer)(x::Union{Real,Missing})
    return (x - m.μ) / m.σ
end

function (m::Normalizer)(x::AbstractVector)
    return (x .- m.μ) ./ m.σ
end

load_data(path) = CSV.File(path) |> DataFrame

function data_splits(df, train_perc)
    train_id = sample(1:nrow(df), Int(floor(train_perc * nrow(df))), replace = false, ordered = false)
    df_train = df[train_id, :]
    df_eval = df[InvertedIndex(train_id), :]
    return df_train, df_eval
end