@info "@__DIR__: " @__DIR__
@info "readdir(@__DIR__): " readdir(@__DIR__)

@info "Initializing packages"
using ScoringEngineDemo
using HTTP
using Sockets
using JSON3
using JSONTables
using DataFrames

@info "Initializing scoring service"
function welcome(req::HTTP.Request)
    return HTTP.Response(200, JSON3.write("Service is active"))
end

function score_flux(req::HTTP.Request)
    df = JSON3.read(IOBuffer(HTTP.payload(req))) |> jsontable |> DataFrame
    score = ScoringEngineDemo.infer_flux(df)
    res = Dict(:score => score)
    return HTTP.Response(200, JSON3.write(res))
end

function score_gbt(req::HTTP.Request)
    df = JSON3.read(IOBuffer(HTTP.payload(req))) |> jsontable |> DataFrame
    score = ScoringEngineDemo.infer_gbt(df)
    res = Dict(:score => score)
    return HTTP.Response(200, JSON3.write(res))
end

# define REST endpoints to dispatch to "service" functions
const SCORING_ROUTER = HTTP.Router()

HTTP.@register(SCORING_ROUTER, "GET", "/", welcome)
HTTP.@register(SCORING_ROUTER, "POST", "/api/v1/flux", score_flux)
HTTP.@register(SCORING_ROUTER, "POST", "/api/v1/gbt", score_gbt)

@info "Ready ▷"
HTTP.serve(SCORING_ROUTER, ip"0.0.0.0", 8008)