"""
    get_shap_importance(df_shap)
Returns feature importance dataframe from a SHAP run result.
"""
function get_shap_importance(df_shap)
    dfg = groupby(df_shap, :feature_name)
    df = combine(dfg, :shap_effect => (x -> mean(abs.(x))) => :shap_effect)
    sort!(df, :shap_effect, rev=false)
    return df
end

"""
    plot_shap_importance(df; model, target_features)
Returns the ploting elements (traces, layout, config) for feature importance
"""
function plot_shap_importance(df; color, title="")

    trace = PlotlyBase.bar(
        x=df[:, :shap_effect],
        y=df[:, :feature_name],
        name="Feature importance",
        orientation="h",
        marker=attr(color=color, size=5, opacity=1.0))

    p_layout = PlotlyBase.Layout(
        plot_bgcolor="white",
        title=title,
        xaxis=attr(linecolor="black", gridcolor="lightgray"),
        yaxis=attr(linecolor="black", gridcolor="none"),
        height="auto")

    p_config = PlotlyBase.PlotConfig()


    return (
        traces=[trace],
        layout=p_layout,
        config=p_config,
    )
end


"""
    get_shap_effect(df_shap; feat)
"""
function get_shap_effect(df_shap; feat)
    df = df_shap[df_shap.feature_name.==feat, :]
    transform!(df, :feature_value => ByRow(x -> convert(Float64, x)) => :feature_value)
    model = loess(df[:, :feature_value], df[:, :shap_effect], span=1.0)
    smooth_x = range(extrema(df[:, :feature_value])...; length=10)
    smooth_y = Loess.predict(model, smooth_x)
    return (x=df[:, :feature_value], y=df[:, :shap_effect], smooth_x=smooth_x, smooth_y=smooth_y)
end


"""
    plot_shap_effect(df; model, target_features)
Returns the ploting elements (traces, layout, config) for feature importance
"""
function plot_shap_effect(shap_effect; color, title="", name="")

    traces = GenericTrace[]

    push!(traces, PlotlyBase.scatter(
        x=shap_effect[:x],
        y=shap_effect[:y],
        mode="markers",
        name="$name",
        marker=attr(color=color, size=5, opacity=1.0))
    )

    push!(traces, PlotlyBase.scatter(
        x=shap_effect[:smooth_x],
        y=shap_effect[:smooth_y],
        name="$name smoothed",
        mode="lines",
        marker=attr(color=color, size=5, opacity=1.0))
    )

    p_layout = PlotlyBase.Layout(
        plot_bgcolor="white",
        title=title,
        xaxis=attr(linecolor="black", gridcolor="lightgray"),
        yaxis=attr(linecolor="black", gridcolor="lightgray"),
        legend=attr(orientation="h"),
        autosize=true,
        height="auto")

    p_config = PlotlyBase.PlotConfig()

    return (
        traces=traces,
        layout=p_layout,
        config=p_config,
    )
end




"""
    get_shap_importance(df_shap)
Returns feature importance dataframe from a SHAP run result.
"""
function get_shap_explain(df_shap)
    df = transform(df_shap, :shap_effect => ByRow(abs) => :shap_effect_abs)
    sort!(df, :shap_effect_abs, rev=true)
    return df
end

"""
    plot_shap_effect(df; model, target_features)
Returns the ploting elements (traces, layout, config) for feature importance
"""
function plot_shap_explain(df_expain; title="")

    traces = GenericTrace[]

    push!(traces, PlotlyBase.waterfall(
        x=df_expain[:, :shap_effect],
        y=df_expain[:, :feature_name],
        text=df_expain[:, :feature_value],
        orientation="h",
        decreasing=attr(marker=attr(color=j_red)),
        increasing=attr(marker=attr(color=j_blue)),
    )
    )

    p_layout = PlotlyBase.Layout(
        plot_bgcolor="white",
        title=title,
        xaxis=attr(linecolor="black"),
        yaxis=attr(autorange="reversed", linecolor="black"),
        autosize=true,
        height="auto")

    p_config = PlotlyBase.PlotConfig()

    return (
        traces=traces,
        layout=p_layout,
        config=p_config,
    )
end