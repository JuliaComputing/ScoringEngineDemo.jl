"""
binvar: cut a variable into nbins
"""
function binvar(x, nbins; method="quantiles")

    if method == "quantiles"
        edges = quantile(x, (1:nbins) / nbins)
        if length(edges) == 0
            edges = [minimum(x)]
        end

    elseif method == "linear"
        edges = range(minimum(x), maximum(x), nbins + 1)
    end
    x_bin = searchsortedlast.(Ref(edges[1:end-1]), x) .+ 1
    return x_bin
end

"""
    one_way_data(df, groupvar, targets, nbins; method="quantiles")

Data prep for one-way plots
"""
function one_way_data(df, groupvar, nbins=10; targets, method="quantiles")
    isa(targets, String) ? targets = [targets] : nothing
    dfs = DataFrames.select(df, groupvar => (x -> binvar(x, nbins; method)) => "var_bins", targets)
    dfg = groupby(dfs, "var_bins")
    dfg = combine(dfg, targets .=> mean .=> targets, targets[1] => length => "_weight")
    return dfg
end

"""
    one_way_plot(df)

One-way plot
"""
const color_dict = Dict(
    "event" => j_red,
    "flux" => j_green,
    "gbt" => j_purple)

function one_way_plot(df; targets)

    isa(targets, String) ? targets = [targets] : nothing
    traces = GenericTrace[]

    for target in targets
        color = get(color_dict, target, "blue")
        push!(traces,
            PlotlyBase.scatter(
                x=df[:, :var_bins],
                y=df[:, target],
                mode="markers+lines",
                name=target,
                marker=attr(color=color, size=5, opacity=1.0)))
    end

    p_layout = PlotlyBase.Layout(
        # width=800,
        # height=500,
        # yaxis_range=[0, max_y],
        title="One-way effect",
        plot_bgcolor="white",
        xaxis=attr(title="Feature value"),
        yaxis=attr(title="Target value", linecolor="black", gridcolor="lightgray"),
        legend=attr(orientation="h")
        # margin=attr(l=0, t=50, r=50, b=0),
        # autosize=true,
    )

    p_config = PlotlyBase.PlotConfig(displaylogo=false, displayModeBar=false, responsive=false)

    return (
        traces=traces,
        layout=p_layout,
        config=p_config,
    )
end

"""
    one_way_plot(df)

One-way plot over weight histogram
"""
function one_way_plot_weights(df; targets)

    isa(targets, String) ? targets = [targets] : nothing
    traces = GenericTrace[]

    push!(traces,
        PlotlyBase.bar(
            x=df[:, :var_bins],
            y=df[:, :_weight],
            name="Weights",
            marker=attr(color="gray", size=5, opacity=0.5)))

    for target in targets
        color = get(color_dict, target, "blue")
        push!(traces,
            PlotlyBase.scatter(
                x=df[:, :var_bins],
                y=df[:, target],
                mode="markers+lines",
                name=target,
                yaxis="y2",
                marker=attr(color=color, size=5, opacity=1.0)))
    end

    p_layout = PlotlyBase.Layout(
        # width=800,
        # height=500,
        # yaxis_range=[0, max_y],
        title="One-way effect",
        plot_bgcolor="white",
        xaxis=attr(title="Feature groups"),
        yaxis=attr(title="Weight", side="right"),
        yaxis2=attr(title="Target value", linecolor="none", gridcolor="lightgray", overlaying="y", side="left"),
        # margin=attr(l=0, t=0, r=0, b=0),
        legend=attr(orientation="h"),
        autosize=true,
    )

    p_config = PlotlyBase.PlotConfig(displaylogo=false, displayModeBar=false, responsive=false)

    return (
        traces=traces,
        layout=p_layout,
        config=p_config,
    )
end