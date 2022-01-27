using CSV
using DataFrames
using Statistics: mean, std
using StatsBase: sample
using BSON

function build_preproc(df; norm_feats)

    df_fit = copy(df) # shoud not mutate the input df
    preproc = Preproc([])

    # density transformation
    push!(preproc.layers, ["population", "town_surface_area"] => density => "density")
    push!(preproc.layers, "density" => (x -> log.(max.(x, 0.01))) => "density")
    df_fit = preproc(df_fit, 1:2)

    push!(preproc.layers, cov_mapping)
    df_fit = preproc(df_fit, 3)

    push!(preproc.layers, drv_sex1)
    df_fit = preproc(df_fit, 4)

    push!(preproc.layers, drv_exp_yrs)
    df_fit = preproc(df_fit, 5)

    push!(preproc.layers, has_drv2)
    df_fit = preproc(df_fit, 6)

    push!(preproc.layers, is_drv2_male)
    df_fit = preproc(df_fit, 7)

    # normalise features
    norms = [feat => Normalizer(df_fit[:, feat]) => feat for feat in norm_feats]
    push!(preproc.layers, norms...)
    df_fit = preproc(df_fit, length(preproc.layers)-length(norms)+1:length(preproc.layers))

    # handle missing values
    miss_handling = norm_feats .=> (x -> coalesce.(x, 0.0)) .=> norm_feats
    push!(preproc.layers, miss_handling...)
    df_fit = preproc(df_fit, length(preproc.layers)-length(miss_handling)+1:length(preproc.layers))

    return preproc
end