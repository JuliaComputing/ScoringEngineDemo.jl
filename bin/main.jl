@info "Julia version: " VERSION
@info "@__DIR__: " @__DIR__
@info "readdir(@__DIR__): " readdir(@__DIR__)

ENV["GENIE_ENV"] = "dev"

@info "Initializing packages"
using Revise
using ScoringEngineDemo
using BSON
using HTTP
using Sockets
using JSON3
using DataFrames
using Stipple
using StippleUI
using StipplePlotly
using PlotlyBase
using Random

using ShapML
using Weave

using StatsBase: sample
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

const df_preds = begin
    df_preds = copy(df_tot)
    add_scores!(df_preds)
    df_preds
end

const years = unique(df_tot[!, "year"])
const rng = Random.MersenneTwister(123)


const df_sample = begin
    sample_size = 30
    ids = sample(rng, 1:nrow(df_tot), sample_size, replace=false, ordered=true)
    df_tot[ids, :]
end

const p_importance_flux = begin
    df_shap = run_shap(df_sample, model="flux", target_features=features_importance)
    df_importance = get_shap_importance(df_shap)
    plot_shap_importance(df_importance, color=j_green, title="Flux feature importance")
end

const p_importance_gbt = begin
    df_shap = run_shap(df_sample, model="gbt", target_features=features_importance)
    df_importance = get_shap_importance(df_shap)
    plot_shap_importance(df_importance, color=j_purple, title="GBT feature importance")
end

####################################
# Application 
####################################

@reactive mutable struct Model <: ReactiveModel

    # features
    features::R{Vector{String}} = features_effect
    feature::R{String} = "vh_value"

    # One-way plots
    groupmethod::R{String} = "quantiles"
    one_way_traces::R{Vector{GenericTrace}} = [PlotlyBase.scatter()]
    one_way_layout::R{PlotlyBase.Layout} = PlotlyBase.Layout()
    one_way_config::R{PlotlyBase.PlotConfig} = PlotlyBase.PlotConfig()

    # plot_layout and config: Plotly specific 
    shap_effect_traces::R{Vector{GenericTrace}} = [PlotlyBase.scatter()]
    shap_effect_layout::R{PlotlyBase.Layout} = PlotlyBase.Layout()
    shap_effect_config::R{PlotlyBase.PlotConfig} = PlotlyBase.PlotConfig()

    # Flux feature importance
    explain_flux_traces::R{Vector{GenericTrace}} = [PlotlyBase.scatter()]
    explain_flux_layout::R{PlotlyBase.Layout} = PlotlyBase.Layout()
    explain_flux_config::R{PlotlyBase.PlotConfig} = PlotlyBase.PlotConfig()

    # GBT feature importance
    explain_gbt_traces::R{Vector{GenericTrace}} = [PlotlyBase.scatter()]
    explain_gbt_layout::R{PlotlyBase.Layout} = PlotlyBase.Layout()
    explain_gbt_config::R{PlotlyBase.PlotConfig} = PlotlyBase.PlotConfig()

    # Flux feature importance
    hist_flux_traces::R{Vector{GenericTrace}} = p_importance_flux[:traces]
    hist_flux_layout::R{PlotlyBase.Layout} = p_importance_flux[:layout]
    hist_flux_config::R{PlotlyBase.PlotConfig} = p_importance_flux[:config]

    # GBT feature importance
    hist_gbt_traces::R{Vector{GenericTrace}} = p_importance_gbt[:traces]
    hist_gbt_layout::R{PlotlyBase.Layout} = p_importance_gbt[:layout]
    hist_gbt_config::R{PlotlyBase.PlotConfig} = p_importance_gbt[:config]

    weave::R{Bool} = false
    resample::R{Bool} = false
end


"""
    one_way_plot!(df, m::Model)

Return one-way effect plot based on selected var
"""
function one_way_plot!(df, m::Model)

    targets = ["event", "flux", "gbt"]
    df_bins = ScoringEngineDemo.one_way_data(df, m.feature[], 10; targets, method=m.groupmethod[])
    p = ScoringEngineDemo.one_way_plot_weights(df_bins; targets)

    m.one_way_traces[] = p[:traces]
    m.one_way_layout[] = p[:layout]
    m.one_way_config[] = p[:config]

    return nothing
end


function shap_effect_plot!(df, m::Model)

    sample_size = 50
    ids = sample(rng, 1:nrow(df), sample_size, replace=false, ordered=true)
    df_sample = df[ids, :]

    df_shap_flux = run_shap(df_sample, model="flux"; reference=df, target_features=[m.feature[]])
    shap_effect_flux = get_shap_effect(df_shap_flux, feat=m.feature[])

    df_shap_gbt = run_shap(df_sample, model="gbt"; reference=df, target_features=[m.feature[]])
    shap_effect_gbt = get_shap_effect(df_shap_gbt, feat=m.feature[])

    p_flux = plot_shap_effect(shap_effect_flux, color=j_green, title="Feature effect", name="flux")
    p_gbt = plot_shap_effect(shap_effect_gbt, color=j_purple, title="Feature effect", name="gbt")

    m.shap_effect_traces[] = [p_flux[:traces]..., p_gbt[:traces]...]
    m.shap_effect_layout[] = p_flux[:layout]
    m.shap_effect_config[] = p_flux[:config]

    return nothing
end

function shap_explain_plot!(df, m::Model)

    sample_size = 1
    ids = sample(rng, 1:nrow(df), sample_size, replace=false, ordered=true)
    df_sample = df[ids, :]

    df_shap_flux = run_shap(df_sample, model="flux"; reference=df, target_features=features_importance)
    df_explain_flux = get_shap_explain(df_shap_flux)

    df_shap_gbt = run_shap(df_sample, model="gbt"; reference=df, target_features=features_importance)
    df_explain_gbt = get_shap_explain(df_shap_gbt)

    p_flux = plot_shap_explain(df_explain_flux, title="Flux explain")
    p_gbt = plot_shap_explain(df_explain_gbt, title="GBT explain")

    m.explain_flux_traces[] = p_flux[:traces]
    m.explain_flux_layout[] = p_flux[:layout]
    m.explain_flux_config[] = p_flux[:config]

    m.explain_gbt_traces[] = p_gbt[:traces]
    m.explain_gbt_layout[] = p_gbt[:layout]
    m.explain_gbt_config[] = p_gbt[:config]

    return nothing
end

"""
    prepare_report(df, model::Model)
WIP: Report generation with Weave
"""
function prepare_report(df, model::Model)

    isempty(model.feature[]) && return nothing
    ids = sample(1:nrow(df), sample_size, replace=false, ordered=true)
    df_sample = df[ids, :]
    @info "df size" size(df_sample)
    add_scores!(df_sample)
    data_flux = run_shap_flux(df_sample, "flux")
    feat_flux = get_feat_importance(data_flux)
    shap_flux = plot_shap(data_flux, model.feature[])

    data_gbt = run_shap_gbt(df_sample, "gbt")
    feat_gbt = get_feat_importance(data_gbt)
    shap_gbt = plot_shap(data_gbt, model.feature[])

    data = Dict(
        :shap_flux => shap_flux,
        :shap_gbt => shap_gbt,
        :feat_flux => feat_flux,
        :feat_gbt => feat_gbt)

    return data
end

function handlers(model::Model)

    on(model.isready) do _
        one_way_plot!(df_preds, model)
        shap_effect_plot!(df_tot, model)
        shap_explain_plot!(df_tot, model)
    end

    on(model.feature) do _
        one_way_plot!(df_preds, model)
        shap_effect_plot!(df_tot, model)
    end

    on(model.groupmethod) do _
        one_way_plot!(df_preds, model)
    end

    on(model.resample) do _
        if model.resample[]
            shap_effect_plot!(df_tot, model)
            shap_explain_plot!(df_tot, model)
            model.resample[] = false
        end
    end

    # on(model.weave) do _
    #     if (model.weave[])
    #         data = prepare_report(df_tot, model)
    #         weave("report.jmd",
    #             doctype="pandoc2html",
    #             pandoc_options=["--toc", "--toc-depth= 3", "--self-contained"],
    #             out_path=@__DIR__,
    #             fig_path=joinpath(@__DIR__, "fig"),
    #             args=data)
    #         model.weave[] = false
    #     end
    # end

    return model
end

include("ui.jl")

route("/") do
    Model |> init |> handlers |> ui |> html
end

Stipple.Genie.startup(8000, "0.0.0.0", async=false)
# down()