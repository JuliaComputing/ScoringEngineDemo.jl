@info "@__DIR__: " @__DIR__
@info "readdir(@__DIR__): " readdir(@__DIR__)

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

const assets_path = joinpath(pkgdir(ScoringEngineDemo), "assets")
const preproc_flux = BSON.load(joinpath(assets_path, "preproc-flux.bson"), ScoringEngineDemo)[:preproc]
const preproc_gbt = BSON.load(joinpath(assets_path, "preproc-gbt.bson"), ScoringEngineDemo)[:preproc]

const adapter_flux = BSON.load(joinpath(assets_path, "adapter-flux.bson"), ScoringEngineDemo)[:adapter]
const adapter_gbt = BSON.load(joinpath(assets_path, "adapter-gbt.bson"), ScoringEngineDemo)[:adapter]

const model_flux = BSON.load(joinpath(assets_path, "model-flux.bson"), ScoringEngineDemo)[:model]
const model_gbt = BSON.load(joinpath(assets_path, "model-gbt.bson"), ScoringEngineDemo)[:model]

@info "Initializing scoring service"
function welcome(req::HTTP.Request)
    return HTTP.Response(200, JSON3.write("Service is active"))
end

function score_flux(req::HTTP.Request)
    df = JSON3.read(IOBuffer(HTTP.payload(req))) |> jsontable |> DataFrame
    score = df |> preproc_flux |> adapter_flux |> model_flux |> logit
    res = Dict(:score => score)
    return HTTP.Response(200, JSON3.write(res))
end

function score_gbt(req::HTTP.Request)
    df = JSON3.read(IOBuffer(HTTP.payload(req))) |> jsontable |> DataFrame
    score = ScoringEngineDemo.predict(model_gbt, df |> preproc_gbt |> adapter_gbt)
    res = Dict(:score => score)
    return HTTP.Response(200, JSON3.write(res))
end

# define REST endpoints to dispatch to "service" functions
const SCORING_ROUTER = HTTP.Router()

HTTP.@register(SCORING_ROUTER, "GET", "/", welcome)
HTTP.@register(SCORING_ROUTER, "POST", "/api/v1/flux", score_flux)
HTTP.@register(SCORING_ROUTER, "POST", "/api/v1/gbt", score_gbt)

@info "Ready ▷"
# HTTP.serve(SCORING_ROUTER, ip"0.0.0.0", 8008)
HTTP.serve(SCORING_ROUTER, ip"127.0.0.1", 8008)