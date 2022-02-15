using HTTP
using Flux
using Sockets
using JSON3


# "service" functions to actually do the work
function score_get(req::HTTP.Request)
    input = HTTP.queryparams(HTTP.URI(req.target))
    res = Dict(:target_uri => req.target, params => input)
    return HTTP.Response(200, JSON3.write(res))
end

function score_post(req::HTTP.Request)
    input = JSON3.read(IOBuffer(HTTP.payload(req)), Dict)
    return HTTP.Response(200, JSON3.write(input))
end

# define REST endpoints to dispatch to "service" functions
const SCORING_ROUTER = HTTP.Router()

HTTP.@register(SCORING_ROUTER, "GET", "/api/v1/risk", score_get)
HTTP.@register(SCORING_ROUTER, "POST", "/api/v1/risk", score_post)
HTTP.serve(SCORING_ROUTER, ip"127.0.0.1", 8005)


############################################
# Querying the api
############################################

# GET request
req = HTTP.request("GET", "http://127.0.0.1:8005/api/v1/risk?param1=2.32&param2=-5.76")
j = JSON3.read(req.body, Dict)

# Alternative GET synthax
req = HTTP.request("GET", "http://127.0.0.1:8005/api/v1/risk", query = Dict(:param1 => 2.32, :param2 => -5.76))

# POST request
body =  Dict(:param1 => 2.99, :param2 => -5.99)
req = HTTP.request("POST", "http://127.0.0.1:8005/api/v1/risk", [], JSON3.write(body))