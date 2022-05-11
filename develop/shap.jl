@info "Initializing packages"
using ScoringEngineDemo
using BSON
using HTTP
using Sockets
using JSON3
using DataFrames
using PlotlyBase
using Random

using ShapML
using Loess

using StatsBase: sample, quantile
using Statistics: mean, std

@info "Initializing assets"
@info "pkgdir(ScoringEngineDemo): " pkgdir(ScoringEngineDemo)

sample_size = 21

const j_blue = "#4063D8"
const j_green = "#389826"
const j_purple = "#9558B2"
const j_red = "#CB3C33"

const assets_path = joinpath(pkgdir(ScoringEngineDemo), "assets")

# const df_tot = ScoringEngineDemo.load_data("assets/training_data.csv")
df_tot = begin
    df_tot = ScoringEngineDemo.load_data(joinpath(assets_path, "training_data.csv"))
    dropmissing!(df_tot)
end

const preproc_flux = BSON.load(joinpath(assets_path, "preproc-flux.bson"), ScoringEngineDemo)[:preproc]
const preproc_gbt = BSON.load(joinpath(assets_path, "preproc-gbt.bson"), ScoringEngineDemo)[:preproc]

const adapter_flux = BSON.load(joinpath(assets_path, "adapter-flux.bson"), ScoringEngineDemo)[:adapter]
const adapter_gbt = BSON.load(joinpath(assets_path, "adapter-gbt.bson"), ScoringEngineDemo)[:adapter]

const model_flux = BSON.load(joinpath(assets_path, "model-flux.bson"), ScoringEngineDemo)[:model]
const model_gbt = BSON.load(joinpath(assets_path, "model-gbt.bson"), ScoringEngineDemo)[:model]

@info "Initializing scoring service"
function infer_flux(df::DataFrame)
    score = df |> preproc_flux |> adapter_flux |> model_flux |> logit
    return Float64.(score)
end
function infer_gbt(df::DataFrame)
    score = ScoringEngineDemo.predict(model_gbt, df |> preproc_gbt |> adapter_gbt) |> vec
    return Float64.(score)
end

# inference on preprocessed df
function infer_flux_post(df::DataFrame)
    score = df |> adapter_flux |> model_flux |> logit
    return Float64.(score)
end
function infer_gbt_post(df::DataFrame)
    score = ScoringEngineDemo.predict(model_gbt, df |> adapter_gbt) |> vec
    return Float64.(score)
end

# features to be passed to shap function
const features_importance = ["pol_no_claims_discount", "pol_coverage", "pol_duration",
    "pol_sit_duration", "vh_value", "vh_weight", "vh_age", "population",
    "town_surface_area", "drv_sex1", "drv_age1", "pol_pay_freq"]

# available features for one-way effect - only numeric features ATM
const features_effect = ["pol_no_claims_discount", "pol_duration", "pol_sit_duration", "vh_value",
    "vh_weight", "vh_age", "population", "town_surface_area", "drv_age1"]

function add_scores!(df::DataFrame)
    scores_flux = infer_flux(df)
    scores_gbt = infer_gbt(df)
    df[:, :flux] .= scores_flux
    df[:, :gbt] .= scores_gbt
    return nothing
end

function pred_shap_flux(model, df)
    scores = infer_flux(df::DataFrame)
    pred_df = DataFrame(score=scores)
    return pred_df
end

function pred_shap_gbt(model, df)
    scores = infer_gbt(df::DataFrame)
    pred_df = DataFrame(score=scores)
    return pred_df
end

# model argment = infer_flux / infer_gbt
function run_shap_flux(df, model="flux")
    data_shap = ShapML.shap(
        explain=copy(df),
        reference=copy(df),
        target_features=features_importance,
        model="flux",
        predict_function=pred_shap_flux,
        sample_size=sample_size,
        seed=123)
    return data_shap
end

function run_shap_gbt(df, model="gbt")
    data_shap = ShapML.shap(
        explain=copy(df),
        reference=copy(df),
        target_features=features_importance,
        model="gbt",
        predict_function=pred_shap_gbt,
        sample_size=sample_size,
        seed=123)
    return data_shap
end

function plot_shap(data_shap, feat="vh_age")
    df = data_shap[data_shap.feature_name.==feat, :]
    transform!(df, :feature_value => ByRow(x -> convert(Float64, x)) => :feature_value)
    model = loess(df[:, :feature_value], df[:, :shap_effect], span=1.0)
    smooth_x = range(extrema(df[:, :feature_value])...; length=10)
    smooth_y = Loess.predict(model, smooth_x)
    return (df=df, smooth_x=smooth_x, smooth_y=smooth_y)
end

function get_feat_importance(data_shap)
    dfg = groupby(data_shap, :feature_name)
    df = combine(dfg, :shap_effect => (x -> mean(abs.(x))) => :shap_effect)
    sort!(df, :shap_effect, rev=false)
    return df
end

const years = unique(df_tot[!, "year"])
rng = Random.MersenneTwister(123)

ids = sample(1:nrow(df_tot), sample_size, replace=false, ordered=true)
df_sample = df_tot[ids, :]
@info "df size" size(df_sample)
add_scores!(df_sample)
data_flux = run_shap_flux(df_sample, "flux")
feat_flux = get_feat_importance(data_flux)
shap_flux = plot_shap(data_flux, model.feature[])
# shap_flux = plot_shap(data_flux, "vh_weight")

data_gbt = run_shap_gbt(df_sample, "gbt")
feat_gbt = get_feat_importance(data_gbt)
shap_gbt = plot_shap(data_gbt, model.feature[])


####
using BenchmarkTools

sample_size = 30
explain_size = 40
reference_size = 80
ids_explain = sample(1:nrow(df_tot), explain_size, replace=false, ordered=true)
ids_reference = sample(1:nrow(df_tot), reference_size, replace=false, ordered=true)

df_preds = copy(df_tot)
@time add_scores!(df_preds)


"""
binvar: cut a variable into nbins
"""
function binvar(x, nbins; method="quantiles")

    if method == "quantiles"
        edges = quantile(x, (1:nbins) / nbins)
        println(edges)
        if length(edges) == 0
            edges = [minimum(x)]
        end

    elseif method == "linear"
        edges = range(minimum(x), maximum(x), nbins + 1)
        println(edges)
    end
    x_bin = searchsortedlast.(Ref(edges[1:end-1]), x) .+ 1
    return x_bin
end

"""
    One-way plot
"""
function one_way_data(df, groupvar, targets, nbins; method="quantiles")
    targets = isa(targets, String) && [targets]
    dfs = DataFrames.select(df, groupvar => (x -> binvar(x, nbins; method)) => "var_bins", "flux")
    dfg = groupby(dfs, "var_bins")
    dfg = combine(dfg, targets .=> mean .=> targets, targets[1] => sum => "_weight")
    return dfg
end

method = "quantiles"
df_bins = one_way_data(df_preds, "drv_age1", "flux", 10; method);
df_bins = one_way_data(df_preds, "vh_value", "flux", 10; method)
df_bins = one_way_data(df_preds, "vh_age", "flux", 10; method)
df_bins = one_way_data(df_preds, "vh_weight", "flux", 10; method)
# df_bins = one_way_data(df_preds, "pol_coverage", "flux", 10; method)

one_way_plot(df_bins)
one_way_plot_weights(df_bins)

"""
    One-way plot
"""
function one_way_plot(df)

    trace_1 = PlotlyBase.scatter(
        x=df[:, :var_bins],
        y=df[:, :flux],
        mode="markers+lines",
        name="Mean target",
        marker=attr(color=j_blue, size=5, opacity=1.0))

    p_layout = PlotlyBase.Layout(
        width=800,
        height=500,
        # yaxis_range=[0, max_y],
        title="Title",
        plot_bgcolor="white",
        xaxis=attr(title="Feature value"),
        yaxis=attr(title="SHAP effect", linecolor="black", gridcolor="lightgray", overlaying="y", side="left"),
        legend=attr(orientation="h"),
        # margin=attr(l=0, t=50, r=50, b=0),
        autosize=true)

    p_config = PlotlyBase.PlotConfig(displaylogo=false, displayModeBar=false, responsive=false)

    p = PlotlyBase.Plot(
        [trace_1],
        p_layout;
        config=p_config)

    return p
end

function one_way_plot_weights(df)

    trace_1 = PlotlyBase.bar(
        x=df[:, :var_bins],
        y=df[:, :_weight],
        name="Weights",
        marker=attr(color="gray", size=5, opacity=0.5))

    trace_2 = PlotlyBase.scatter(
        x=df[:, :var_bins],
        y=df[:, :flux],
        mode="markers+lines",
        name="trace A",
        yaxis="y2",
        marker=attr(color=j_blue, size=5, opacity=1.0))

    p_layout = PlotlyBase.Layout(
        width=800,
        height=500,
        # yaxis_range=[0, max_y],
        title="Title",
        plot_bgcolor="white",
        xaxis=attr(title="Feature value"),
        yaxis=attr(title="Weight", side="right"),
        yaxis2=attr(title="SHAP effect", linecolor="black", gridcolor="lightgray", overlaying="y", side="left"),
        # margin=attr(l=0, t=50, r=50, b=0),
        legend=attr(orientation="h"),
        autosize=true)

    p_config = PlotlyBase.PlotConfig(displaylogo=false, displayModeBar=false, responsive=false)

    p = PlotlyBase.Plot(
        [trace_1],
        p_layout;
        config=p_config)

    p = PlotlyBase.Plot(
        [trace_1, trace_2],
        p_layout;
        config=p_config)

    return p
end

df_explain = df_tot[ids_explain, :]
df_reference = df_tot[ids_reference, :]
target_features = features_importance[1:12]

# parallel: :none, :samples (), :features (), :both ()]
# 401.203 ms for all
@time data_shap_gbt_1 = ShapML.shap(
    explain=df_explain,
    reference=df_reference,
    target_features=target_features,
    model="gbt",
    predict_function=pred_shap_gbt,
    sample_size=sample_size,
    parallel=:samples,
    seed=123);

df_test_1 = subset(data_shap_gbt_1, :feature_name => x -> x .== "pol_no_claims_discount")
df_test_2 = subset(data_shap_gbt_1, :feature_name => x -> x .== "pol_coverage")

sort!(df_test_1, :feature_value)

trace_1 = PlotlyBase.scatter(
    x=df_test_1[:, :feature_value],
    y=df_test_1[:, :shap_effect],
    mode="markers+lines",
    name="trace A",
    marker=attr(size=5, opacity=0.7))

trace_2 = PlotlyBase.scatter(
    x=df_test_1[:, :feature_value] .+ 0.01,
    y=df_test_1[:, :shap_effect],
    mode="markers+lines",
    name="trace BB",
    marker=attr(size=5, opacity=0.7))

p_layout = PlotlyBase.Layout(width=800,
    height=500,
    # yaxis_range=[0, max_y],
    title="Title",
    plot_bgcolor="white",
    yaxis=attr(title="SHAP effect", linecolor="black", gridcolor="lightgray"),
    xaxis=attr(title="Feature value"))

p_config = PlotlyBase.PlotConfig(displaylogo=false, displayModeBar=false, responsive=false)

p = PlotlyBase.Plot(
    [trace_1, trace_2],
    p_layout;
    config=p_config)

str = JSON.json(p)
contains(str, "displayModeBar")
x1 = match(r"displaylogo", str)
str[x1.offset:x1.offset+100]

@time data_shap_flux_1 = ShapML.shap(
    explain=df_explain,
    reference=df_reference,
    target_features=target_features,
    model="flux",
    predict_function=pred_shap_flux,
    sample_size=sample_size,
    parallel=:samples,
    seed=123);
