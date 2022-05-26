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
    sample_size::R{Int} = 50
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

    ids = sample(rng, 1:nrow(df), m.sample_size[], replace=false, ordered=true)
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

    onany(model.sample_size) do (_...)
        shap_effect_plot!(df_tot, model)
        shap_explain_plot!(df_tot, model)
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