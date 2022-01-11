using ScoringEngine
using BSON
using HTTP
using Sockets
using JSON3
using JSONTables
using DataFrames

function render(req::HTTP.Request)
    input = JSON3.read(IOBuffer(HTTP.payload(req))) |> jsontable
    @info "typeof(input)" typeof(input)
    @info "input" input
    df = input |> DataFrame
    @info "DF conversion succeed"
    res = Dict(:nrow => nrow(df))
    return HTTP.Response(200, JSON3.write(res))
end

const SCORING_ROUTER = HTTP.Router()
HTTP.@register(SCORING_ROUTER, "POST", "/", render)
HTTP.serve(SCORING_ROUTER, ip"0.0.0.0", 8008)

using DataFrames
using HTTP
using JSON3
using JSONTables

df = DataFrame(:gfdlj_ghdf_j56_45_dflkj => 1.1, :var_a12_gf_gdf_65 => 2.2)
df = DataFrame(:v1_dfg_lkdfglkj_gfdl => 1.1, :kdfg45_dfg_lkdfglkj_gfdl => 2.2, :sddf_sdf_548gj_9856_6=> 2.2)
body = JSON3.write(arraytable(df))
body = JSON3.write(objecttable(df))

JSON3.read(body) |> jsontable
JSON3.read(body) |> jsontable |> DataFrame
req = HTTP.request("POST", "http://localhost:8008", [], body)
