@info "Initializing packages"
using ScoringEngineDemo
using BSON
using JSON3
using DataFrames
using PlotlyBase
using Random
using ShapML

using StatsBase: sample, quantile
using Statistics: mean, std

@info "Initializing assets"
@info "pkgdir(ScoringEngineDemo): " pkgdir(ScoringEngineDemo)

const j_blue = "#4063D8"
const j_green = "#389826"
const j_purple = "#9558B2"
const j_red = "#CB3C33"

const assets_path = joinpath(pkgdir(ScoringEngineDemo), "assets")

# const df_tot = ScoringEngineDemo.load_data("assets/training_data.csv")
df_tot = begin
    df_tot = ScoringEngineDemo.load_data(joinpath(assets_path, "training_data.csv"))
    transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")
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

function run_shap(df; reference=nothing, model, target_features, sample_size=30)
    if model == "flux"
        predict_function = pred_shap_flux
    elseif model == "gbt"
        predict_function = pred_shap_gbt
    else
        error("Argument `model` needs to be one of: `flux` or `gbt`.")
    end

    isnothing(reference) ? reference = copy(df) : nothing
    data_shap = ShapML.shap(
        explain=df,
        reference=reference,
        target_features=target_features,
        model=model,
        predict_function=predict_function,
        sample_size=sample_size,
        seed=123)
    return data_shap
end

# features to be passed to shap function
const features_importance = ["pol_no_claims_discount", "pol_coverage", "pol_duration",
    "pol_sit_duration", "vh_value", "vh_weight", "vh_age", "population",
    "town_surface_area", "drv_sex1", "drv_sex2", "drv_age1", "pol_pay_freq", "drv_age_lic1"]

# available features for one-way effect - only numeric features ATM
const features_effect = ["pol_no_claims_discount", "pol_duration", "pol_sit_duration", "vh_value",
    "vh_weight", "vh_age", "population", "town_surface_area", "drv_age1", "drv_age_lic1"]

function add_scores!(df::DataFrame)
    scores_flux = infer_flux(df)
    scores_gbt = infer_gbt(df)
    df[:, :flux] .= scores_flux
    df[:, :gbt] .= scores_gbt
    return nothing
end

const years = unique(df_tot[!, "year"])
const rng = Random.MersenneTwister(123)

############################
# SHAP importance
############################
sample_size = 30
explain_size = 50
ids = sample(rng, 1:nrow(df_tot), explain_size, replace=false, ordered=true)
df_sample = df_tot[ids, :]

@time df_shap_flux = run_shap(df_sample, model="flux"; sample_size, reference=df_tot, target_features=features_importance);
df_importance_flux = get_shap_importance(df_shap_flux)

@time df_shap_gbt = run_shap(df_sample, model="gbt"; sample_size, reference=df_tot, target_features=features_importance);
df_importance_gbt = get_shap_importance(df_shap_gbt)

p_importance_flux = plot_shap_importance(df_importance_flux, color=j_purple, title="Flux feature importance");
PlotlyBase.Plot(p_importance_flux[:traces], p_importance_flux[:layout]; config=p_importance_flux[:config])

p_importance_gbt = plot_shap_importance(df_importance_gbt, color=j_purple, title="GBT feature importance");
PlotlyBase.Plot(p_importance_gbt[:traces], p_importance_gbt[:layout]; config=p_importance_gbt[:config])

############################
# SHAP effect
############################
sample_size = 30
explain_size = 50
ids = sample(rng, 1:nrow(df_tot), explain_size, replace=false, ordered=true)
df_sample = df_tot[ids, :]

df_shap_flux = run_shap(df_sample, model="flux"; reference=df_tot, target_features=["vh_age"])
shap_effect_flux = get_shap_effect(df_shap_flux, feat="vh_age")

df_shap_gbt = run_shap(df_sample, model="gbt"; reference=df_tot, target_features=["vh_age"])
shap_effect_gbt = get_shap_effect(df_shap_gbt, feat="vh_age")

p_flux = plot_shap_effect(shap_effect_flux, color=j_green, title="Feature effect", name="flux")
p_gbt = plot_shap_effect(shap_effect_gbt, color=j_purple, title="Feature effect", name="gbt")

traces = [p_flux[:traces]..., p_gbt[:traces]...]
layout = p_flux[:layout]
config = p_flux[:config]
PlotlyBase.Plot(traces, layout; config=config)

############################
# SHAP explain
############################
sample_size = 1
ids = sample(rng, 1:nrow(df_tot), sample_size, replace=false, ordered=true)
df_sample = df_tot[ids, :]

@time df_shap_flux = run_shap(df_sample, model="flux"; reference=df_tot, target_features=features_importance)
df_explain_flux = get_shap_explain(df_shap_flux)

@time df_shap_gbt = run_shap(df_sample, model="gbt"; reference=df_tot, target_features=features_importance)
df_explain_gbt = get_shap_explain(df_shap_gbt)

p_flux = ScoringEngineDemo.plot_shap_explain(df_explain_flux, title="Flux explain", name="flux")
p_gbt = ScoringEngineDemo.plot_shap_explain(df_explain_gbt, title="GBT explain", name="gbt")

PlotlyBase.Plot(p_flux[:traces], p_flux[:layout]; config=p_flux[:config])
PlotlyBase.Plot(p_gbt[:traces], p_gbt[:layout]; config=p_gbt[:config])

############################
# One-way effect
############################
df_preds = copy(df_tot)
add_scores!(df_preds)

method = "quantiles"
targets = ["event", "flux", "gbt"]
df_bins = one_way_data(df_preds, "drv_age1", 10; targets, method);
df_bins = one_way_data(df_preds, "vh_value", 10; targets, method)
df_bins = one_way_data(df_preds, "vh_age", 10; targets, method)
df_bins = one_way_data(df_preds, "vh_weight", 10; targets, method)
# df_bins = one_way_data(df_preds, "pol_coverage", "flux", 10; method)

p = one_way_plot(df_bins; targets);
p = one_way_plot_weights(df_bins; targets);

p = PlotlyBase.Plot(
    p[:traces],
    p[:layout];
    config=p[:config])