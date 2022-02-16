using CSV
using DataFrames
using Statistics: mean, std
using StatsBase: sample
using BSON


load_data(path) = CSV.File(path) |> DataFrame

function data_splits(df, train_perc)
    train_id = sample(1:nrow(df), Int(floor(train_perc * nrow(df))), replace = false, ordered = false)
    df_train = df[train_id, :]
    df_eval = df[InvertedIndex(train_id), :]
    return df_train, df_eval
end

"""
Preproc
Preproc functor. Holds a vector of transform operations
"""
struct Preproc <: Function
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

function build_adapter_flux(featnames, targetname)
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

function build_adapter_gbt(featnames, targetname)
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
Normalizer

Constructor store normalisation parameters based on input vector mean and std. 
Functor apply normalisation parameters to input vector
"""
struct Normalizer{T} <: Function
    μ::T
    σ::T
end

Normalizer(x::AbstractVector) = Normalizer(mean(skipmissing(x)), std(skipmissing(x)))

function (m::Normalizer)(x::Union{Real,Missing})
    return (x - m.μ) / m.σ
end

function (m::Normalizer)(x::AbstractVector)
    return (x .- m.μ) ./ m.σ
end


"""
density(pop, area)
"""
density(pop, area) = pop ./ area


"""
years of experience
"""
age_diff(age1, age2) = age2 .- age1
drv_exp_yrs = ["drv_age_lic1", "drv_age1"] => age_diff => "drv_exp_yrs"

"""
    Categorical Mappings
"""
# coverage maping
pol_cov_dict = Dict{String,Float64}(
    "Min" => 1,
    "Med1" => 2,
    "Med2" => 3,
    "Max" => 4)

pol_cov_map(x) = get(pol_cov_dict, x, 4)
cov_mapping = "pol_coverage" => ByRow(pol_cov_map) => "pol_coverage"

# drv_sex1
drv_sex1_dict = Dict{String,Float64}(
    "M" => 0,
    "F" => 1)
drv_sex1_map(x) = get(drv_sex1_dict, x, 0)
drv_sex1 = "drv_sex1" => ByRow(drv_sex1_map) => "drv_sex1"

# drv_sex2 A
drv_sex2_dict_A = Dict{String,Float64}(
    "0" => 0,
    "M" => 1,
    "F" => 1)
drv_sex2_map_A(x) = get(drv_sex2_dict_A, x, 0)
has_drv2 = "drv_sex2" => ByRow(drv_sex2_map_A) => "has_drv2"

# drv_sex2 B
drv_sex2_dict_B = Dict{String,Float64}(
    "0" => 0,
    "M" => 1,
    "F" => 0)
drv_sex2_map_B(x) = get(drv_sex2_dict_B, x, 0)
is_drv2_male = "drv_sex2" => ByRow(drv_sex2_map_B) => "is_drv2_male"
