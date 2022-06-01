@info "Loading assets"
const assets_path = joinpath(pkgdir(ScoringEngineDemo), "assets")

# const normA = BSON.load(joinpath(assets_path, "normA.bson"), @__MODULE__)[:norm]

function __init__()
    global preproc_flux = BSON.load(joinpath(assets_path, "preproc-flux.bson"), @__MODULE__)[:preproc]
    global preproc_gbt = BSON.load(joinpath(assets_path, "preproc-gbt.bson"), @__MODULE__)[:preproc]

    global adapter_flux = BSON.load(joinpath(assets_path, "adapter-flux.bson"), @__MODULE__)[:adapter]
    global adapter_gbt = BSON.load(joinpath(assets_path, "adapter-gbt.bson"), @__MODULE__)[:adapter]

    global model_flux = BSON.load(joinpath(assets_path, "model-flux.bson"), @__MODULE__)[:model]
    global model_gbt = BSON.load(joinpath(assets_path, "model-gbt.bson"), @__MODULE__)[:model]
end

# @info "Loading preproc"
# const preproc_flux = BSON.load(joinpath(assets_path, "preproc-flux.bson"), @__MODULE__)[:preproc]
# const preproc_gbt = BSON.load(joinpath(assets_path, "preproc-gbt.bson"), @__MODULE__)[:preproc]

# const adapter_flux = BSON.load(joinpath(assets_path, "adapter-flux.bson"), @__MODULE__)[:adapter]
# const adapter_gbt = BSON.load(joinpath(assets_path, "adapter-gbt.bson"), @__MODULE__)[:adapter]

# const model_flux = BSON.load(joinpath(assets_path, "model-flux.bson"), @__MODULE__)[:model]
# const model_gbt = BSON.load(joinpath(assets_path, "model-gbt.bson"), @__MODULE__)[:model]

function infer_flux(df::DataFrame)
    score = df |> preproc_flux |> adapter_flux |> model_flux |> logit
    return Float64.(score)
end
function infer_gbt(df::DataFrame)
    score = EvoTrees.predict(model_gbt, df |> preproc_gbt |> adapter_gbt) |> vec
    return Float64.(score)
end

@info "Loading assets completed"
