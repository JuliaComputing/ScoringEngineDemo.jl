using CSV
using DataFrames
using Statistics: mean, std
using StatsBase: sample
using BSON

function build_preproc(df; norm_feats)

    df_fit = copy(df) # shoud not mutate the input df
    preproc = ScoringEngine.Preproc([])

    # density transformation
    push!(preproc.layers, ["population", "town_surface_area"] => ScoringEngine.density => "density")
    push!(preproc.layers, "density" => (x -> log.(max.(x, 0.01))) => "density")
    df_fit = preproc(df_fit, 1:2)

    # coverage maping
    pol_cov_dict = Dict{String,Float64}(
        "Min" => 1,
        "Med1" => 2,
        "Med2" => 3,
        "Max" => 4)

    pol_cov_map(x) = get(pol_cov_dict, x, 4.0)
    cov_mapping = "pol_coverage" => ByRow(pol_cov_map) => "pol_coverage"
    push!(preproc.layers, cov_mapping)
    df_fit = preproc(df_fit, 3)

    # normalise features
    norms = [feat => ScoringEngine.Normalizer(df_fit[:, feat]) => feat for feat in norm_feats]
    push!(preproc.layers, norms...)
    df_fit = preproc(df_fit, length(preproc.layers)-length(norms)+1:length(preproc.layers))

    # handle missing values
    miss_handling = norm_feats .=> (x -> coalesce.(x, 0.0)) .=> norm_feats
    push!(preproc.layers, miss_handling...)
    df_fit = preproc(df_fit, length(preproc.layers)-length(miss_handling)+1:length(preproc.layers))

    return preproc
end