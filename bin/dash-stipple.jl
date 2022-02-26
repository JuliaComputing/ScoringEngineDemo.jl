@info "@__DIR__: " @__DIR__
@info "readdir(@__DIR__): " readdir(@__DIR__)

@info "Initializing packages"
using ScoringEngineDemo
using BSON
using HTTP
using Sockets
using JSON3
using DataFrames
using Stipple
using StippleUI
using StipplePlotly
using Random

using ShapML
using Loess

using StatsBase: sample
using Statistics: mean, std

@info "Initializing assets"
@info "pkgdir(ScoringEngineDemo): " pkgdir(ScoringEngineDemo)

# const df_tot = ScoringEngineDemo.load_data("assets/training_data.csv")

const df_tot = begin
    df_tot = ScoringEngineDemo.load_data("assets/training_data.csv")
    dropmissing!(df_tot)
end

const sample_size = 20

const assets_path = joinpath(pkgdir(ScoringEngineDemo), "assets")
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
    pred_df = DataFrame(score = scores)
    return pred_df
end

function pred_shap_gbt(model, df)
    scores = infer_gbt(df::DataFrame)
    pred_df = DataFrame(score = scores)
    return pred_df
end

# model argment = infer_flux / infer_gbt
function run_shap_flux(df, model = "flux")
    data_shap = ShapML.shap(
        explain = copy(df),
        reference = copy(df),
        target_features = features_importance,
        model = "flux",
        predict_function = pred_shap_flux,
        sample_size = sample_size,
        seed = 123)
    return data_shap
end

function run_shap_gbt(df, model = "gbt")
    data_shap = ShapML.shap(
        explain = copy(df),
        reference = copy(df),
        target_features = features_importance,
        model = "gbt",
        predict_function = pred_shap_gbt,
        sample_size = sample_size,
        seed = 123)
    return data_shap
end

function plot_shap(data_shap, feat = "vh_age")
    df = data_shap[data_shap.feature_name.==feat, :]
    transform!(df, :feature_value => ByRow(x -> convert(Float64, x)) => :feature_value)
    model = loess(df[:, :feature_value], df[:, :shap_effect], span = 1.0)
    smooth_x = range(extrema(df[:, :feature_value])...; length = 10)
    smooth_y = Loess.predict(model, smooth_x)
    return (df = df, smooth_x = smooth_x, smooth_y = smooth_y)
end

function get_feat_importance(data_shap)
    dfg = groupby(data_shap, :feature_name)
    df = combine(dfg, :shap_effect => (x -> mean(abs.(x))) => :shap_effect)
    sort!(df, :shap_effect, rev = false)
    return df
end

const years = unique(df_tot[!, "year"])
rng = Random.MersenneTwister(123)

####################################
# Application 
####################################

@reactive mutable struct Model <: ReactiveModel

    # data_pagination::DataTablePagination = DataTablePagination(rows_per_page = 50) # DataTable, DataTablePagination are part of StippleUI which helps us set Data Table UI

    # plot_layout and config: Plotly specific 
    plot_data::R{Vector{PlotData}} = PlotData[]   # PlotSeries is structure used to store data
    plot_layout::PlotLayout = PlotLayout()
    plot_config::R{PlotConfig} = PlotConfig(displaylogo = false)

    # plot_layout and config: Plotly specific
    hist_flux_data::R{Vector{PlotData}} = PlotData[]   # PlotSeries is structure used to store data
    hist_flux_layout::PlotLayout = PlotLayout()
    hist_flux_config::R{PlotConfig} = PlotConfig(displaylogo = false)

    # plot_layout and config: Plotly specific
    hist_gbt::R{Vector{PlotData}} = PlotData[]   # PlotSeries is structure used to store data
    hist_gbt_layout::PlotLayout = PlotLayout()
    hist_gbt_config::R{PlotConfig} = PlotConfig(displaylogo = false)

    features::R{Vector{String}} = features_effect #iris dataset have following columns: https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis/data
    feature::R{String} = "vh_value"
    # yfeature::R{String} = iris_features[2]
end

#= Computation =#
function plot_data!(df, model::Model)
    plot_data = PlotData[]
    isempty(model.feature[]) && return nothing

    ids = sample(1:nrow(df), sample_size, replace = false, ordered = true)
    df_sample = df[ids, :]
    @info "df size" size(df_sample)
    add_scores!(df_sample)
    data_flux = run_shap_flux(df_sample, "flux")
    feat_flux = get_feat_importance(data_flux)
    shap_flux = plot_shap(data_flux, model.feature[])
    # shap_flux = plot_shap(data_flux, "vh_weight")

    data_gbt = run_shap_gbt(df_sample, "gbt")
    feat_gbt = get_feat_importance(data_gbt)
    shap_gbt = plot_shap(data_gbt, model.feature[])

    # push!(plot_data, PlotData(x = df[:, model.feature[]], y = df[:, "score"],
    #     plot = "scatter", mode = "markers", marker = PlotDataMarker(color = iris_colors[s]), name = "test"))

    # push!(plot_data, PlotData(x = rand(10), y = rand(10), plot = "scatter", mode = "markers", marker = PlotDataMarker(color = "red"), name = "test"))

    push!(plot_data, PlotData(x = shap_flux[:df][:, :feature_value], y = shap_flux[:df][:, :shap_effect],
        plot = "scatter", mode = "markers",
        marker = PlotDataMarker(color = "red", opacity = 0.4, size = 10),
        name = "flux"))

    push!(plot_data, PlotData(x = collect(shap_flux[:smooth_x]), y = collect(shap_flux[:smooth_y]),
        plot = "scatter", mode = "lines",
        marker = PlotDataMarker(color = "red", opacity = 0.6, size = 10),
        name = "flux"))

    push!(plot_data, PlotData(x = shap_gbt[:df][:, :feature_value], y = shap_gbt[:df][:, :shap_effect],
        plot = "scatter", mode = "markers",
        marker = PlotDataMarker(color = "green", opacity = 0.4, size = 10),
        name = "gbt"))

    push!(plot_data, PlotData(x = collect(shap_gbt[:smooth_x]), y = collect(shap_gbt[:smooth_y]),
        plot = "scatter", mode = "lines",
        marker = PlotDataMarker(color = "green", opacity = 0.6, size = 10),
        name = "gbt"))

    model.plot_data[] = plot_data
    return nothing
end


function ui(model::Model)

    onany(model.feature, model.features) do (_...)
        plot_data!(df_tot, model)
        #   compute_clusters!(model)
    end

    page(
        model,
        class = "container",
        title = "Model Diagnosis",
        head_content = Genie.Assets.favicon_support(),
        prepend = style(
            """
            tr:nth-child(even) {
              background: #F8F8F8 !important;
            }
            .modebar {
              display: none!important;
            }
            .st-module {
              background-color: #FFF;
              border-radius: 2px;
              box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.04);
            }
            .stipple-core .st-module > h5,
            .stipple-core .st-module > h6 {
              border-bottom: 0px !important;
            }
            """
        ),
        [
            heading("Model Diagnosis"),
            row([
                cell(
                    class = "st-module col-sm-3",
                    [
                        h6("Feature"),
                        Stipple.select(:feature; options = :features)
                    ]
                )
            ]),
            row([
                cell(
                    class = "st-module",
                    [
                        h5("One-way effect"),
                        plot(:plot_data, layout = :plot_layout, config = :plot_config)
                    ]
                )
            ])
        ]
    )
end


route("/") do
    Model |> init |> ui |> html
end

up(9000; async = true, server = Stipple.bootstrap())
down()