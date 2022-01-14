@info "Initializing packages"
using ScoringEngineDemo
using BSON
using HTTP
using Sockets
using JSON3
using JSONTables
using DataFrames

@info "Initializing assets"
@info "pkgdir(ScoringEngineDemo): " pkgdir(ScoringEngineDemo)
@info "@__DIR__: " @__DIR__
@info "readdir(@__DIR__): " readdir(@__DIR__)

const assets_path = joinpath(pkgdir(ScoringEngineDemo), "assets")
const preproc_flux = BSON.load(joinpath(assets_path, "preproc.bson"), ScoringEngineDemo)[:preproc]
const preproc_gbt = BSON.load(joinpath(assets_path, "preproc.bson"), ScoringEngineDemo)[:preproc]

const preproc_adapt_flux = BSON.load(joinpath(assets_path, "preproc-adapt-flux.bson"), ScoringEngineDemo)[:preproc_adapt]
const preproc_adapt_gbt = BSON.load(joinpath(assets_path, "preproc-adapt-gbt.bson"), ScoringEngineDemo)[:preproc_adapt]

const model_flux = BSON.load(joinpath(assets_path, "model-flux.bson"), ScoringEngineDemo)[:model]
const model_gbt = BSON.load(joinpath(assets_path, "model-gbt.bson"), ScoringEngineDemo)[:model]

@info "Initializing scoring service"
function welcome(req::HTTP.Request)
    return HTTP.Response(200, JSON3.write("Service is active"))
end

function score_post(req::HTTP.Request)
    df = JSON3.read(IOBuffer(HTTP.payload(req))) |> jsontable |> DataFrame
    infer_flux = df |> preproc_flux |> preproc_adapt_flux |> model_flux |> ScoringEngineDemo.logit
    infer_gbt = ScoringEngineDemo.predict(model_gbt, df |> preproc_gbt |> preproc_adapt_gbt)
    res = Dict(:score_flux => infer_flux, :score_gbt => infer_gbt)
    return HTTP.Response(200, JSON3.write(res))
end

# define REST endpoints to dispatch to "service" functions
const SCORING_ROUTER = HTTP.Router()

HTTP.@register(SCORING_ROUTER, "GET", "/", welcome)
HTTP.@register(SCORING_ROUTER, "POST", "/api/v1/risk", score_post)
# HTTP.@register(SCORING_ROUTER, "GET", "/api/v1/risk", score_get)

@info "Ready ▷"
HTTP.serve(SCORING_ROUTER, ip"0.0.0.0", 8008)