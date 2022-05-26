@info "Julia version: " VERSION
@info "@__DIR__: " @__DIR__
@info "readdir(@__DIR__): " readdir(@__DIR__)

ENV["GENIE_ENV"] = "dev"

@info "Initializing packages"
using Revise
using DataFrames
using Stipple
using StippleUI
using StipplePlotly
using PlotlyBase


####################################
# Stipple Model
####################################
@reactive mutable struct Model <: ReactiveModel
    feature::R{String} = "vh_value"
    sample_size::R{Int} = 30
    resample::R{Bool} = false
end


function handlers(model::Model)

    on(model.isready) do _
    end
    
    onany(model.sample_size, model.resample) do (_...)
        if model.resample[]
            @info "resample triggered"
            model.feature[] = string(rand(100:999))
            model.resample[] = false
        end
    end

    return model
end

function ui(m::Model)
    Stipple.page(
        m,
        class="container",
        title="Stipple App",
        head_content=Genie.Assets.favicon_support(),
        [
            heading("Stipple App"),
            row([
                cell(class="st-module", h5("", @text(:feature))),
                cell(class="st-module", h5("Group Method")),
                cell(class="st-module", h5("Sample size")),
                btn("Resample", @click("resample = true"), color="secondary")
            ]),
            row(
                [
                    cell(class="st-module", h1("Content 1"))
                    cell(class="st-module", h1("Content 2"))
                ]),
        ],
        @iif(:isready)
    )
end

route("/") do
    Model |> init |> handlers |> ui |> html
end

Stipple.Genie.startup(8000, "0.0.0.0", async=false)
# down()