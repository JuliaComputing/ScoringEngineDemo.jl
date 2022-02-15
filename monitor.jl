using ScoringEngineDemo
using HTTP
using JSON3
using JSONTables
using DataFrames
using CairoMakie

df_tot = ScoringEngineDemo.load_data(joinpath(pkgdir(ScoringEngineDemo), "assets", "training_data.csv"))
df = df_tot[1:2, :]
body = JSON3.write(arraytable(df))
JSON3.read(body) |> jsontable |> DataFrame

req = HTTP.request("GET", "http://localhost:8008")
req = HTTP.request("POST", "http://localhost:8008/api/v1/flux", [], body)
req = HTTP.request("POST", "http://localhost:8008/api/v1/gbt", [], body)

req = HTTP.request("GET", "https://6lkz9.apps.staging.juliacomputing.io/")
req = HTTP.request("POST", "https://6lkz9.apps.staging.juliacomputing.io/api/v1/flux", [], body)
req = HTTP.request("POST", "https://6lkz9.apps.staging.juliacomputing.io/api/v1/gbt", [], body)

scores_flux = Float64.(JSON3.read(req.body, Dict)["score_flux"])
scores_gbt = Float64.(JSON3.read(req.body, Dict)["score_gbt"])

scatter(Float32.(scores_flux))

init = [0.0]
p_node = Node(init) 
p_node_B = Node(init)
fig, ax = density(p_node, color = (:slategray, 0.3));
density!(p_node_B, color = (:navy, 0.3))
limits!(ax, -0.2, 0.2, 0, 30)
fig

frames = 1:20
record(fig, "data/min-max-flux.mp4", frames; framerate = 4) do frame
    id = rand(1:nrow(df_tot))
    df_A, df_B = df_tot[id:id, :], df_tot[id:id, :]
    # df_A[1, "drv_age1"] = 20
    # df_B[1, "drv_age1"] = 60
    df_A[1, "pol_coverage"] = "Min"
    df_B[1, "pol_coverage"] = "Max"
    df = vcat(df_A, df_B)
    body = JSON3.write(arraytable(df))
    req = HTTP.request("POST", "http://localhost:8008/api/v1/risk", [], body)
    scores = Float64.(JSON3.read(req.body, Dict)["score_flux"])
    spread = diff(scores)[1]
    p_node[] = push!(p_node[], spread)
    p_node[][1] == 0.0 ? deleteat!(p_node[], 1) : nothing
end

record(fig, "data/flux-gbt.mp4", frames; framerate = 4) do frame
    id = rand(1:nrow(df_tot))
    df = df_tot[id:id, :]
    body = JSON3.write(arraytable(df))
    req = HTTP.request("POST", "http://localhost:8008/api/v1/risk", [], body)
    scores_flux = Float64.(JSON3.read(req.body, Dict)["score_flux"])
    scores_gbt = Float64.(JSON3.read(req.body, Dict)["score_gbt"])
    spread = scores_gbt[1] - scores_flux[1]
    p_node[] = push!(p_node[], spread)
    p_node[][1] == 0.0 ? deleteat!(p_node[], 1) : nothing
end

function monitor_spread()
    spread = 0
    for _ in 1:100
        sleep(0.1)
        id = rand(1:nrow(df_tot))
        df_A, df_B = df_tot[id:id, :], df_tot[id:id, :]
        # df_A[1, "pol_coverage"] = "Min"
        # df_B[1, "pol_coverage"] = "Max"
        df_A[1, "drv_age1"] = 40
        df_B[1, "drv_age1"] = 60
        df = vcat(df_A, df_B)

        body = JSONe3.write(arraytable(df))
        req = HTTP.request("POST", "http://localhost:8008/api/v1/risk", [], body)
        scores = Float64.(JSON3.read(req.body, Dict)["score_flux"])
        spread = diff(scores)[1]

        p_node[] = push!(p_node[], spread)
        p_node[][1] == 0.0 ? deleteat!(p_node[], 1) : nothing
    end
end

monitor_spread()
