@info "Initializing assets"
@info "pkgdir(ScoringEngineDemo): " pkgdir(ScoringEngineDemo)

const j_blue = "#4063D8"
const j_green = "#389826"
const j_purple = "#9558B2"
const j_red = "#CB3C33"

const assets_path = joinpath(pkgdir(ScoringEngineDemo), "assets")

const df_tot = begin
    df_tot = ScoringEngineDemo.load_data(joinpath(assets_path, "training_data.csv"))
    transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")
    dropmissing!(df_tot)
end

# features to be passed to shap function
const features_importance = ["pol_no_claims_discount", "pol_coverage", "pol_duration",
    "pol_sit_duration", "vh_value", "vh_weight", "vh_age", "population",
    "town_surface_area", "drv_sex1", "drv_sex2", "drv_age1", "pol_pay_freq", "drv_age_lic1"]

# available features for one-way effect - only numeric features ATM
const features_effect = ["pol_no_claims_discount", "pol_duration", "pol_sit_duration", "vh_value",
    "vh_weight", "vh_age", "population", "town_surface_area", "drv_age1", "drv_age_lic1"]

function add_scores!(df::DataFrame)
    scores_flux = ScoringEngineDemo.infer_flux(df)
    scores_gbt = ScoringEngineDemo.infer_gbt(df)
    df[:, :flux] .= scores_flux
    df[:, :gbt] .= scores_gbt
    return nothing
end

const df_preds = begin
    df_preds = copy(df_tot)
    add_scores!(df_preds)
    df_preds
end

const years = unique(df_tot[!, "year"])
const rng = Random.MersenneTwister(123)


const df_sample = begin
    sample_size = 30
    ids = sample(rng, 1:nrow(df_tot), sample_size, replace=false, ordered=true)
    df_tot[ids, :]
end

const p_importance_flux = begin
    df_shap = ScoringEngineDemo.run_shap(df_sample, model="flux", target_features=features_importance)
    df_importance = get_shap_importance(df_shap)
    plot_shap_importance(df_importance, color=j_green, title="Flux feature importance")
end

const p_importance_gbt = begin
    df_shap = ScoringEngineDemo.run_shap(df_sample, model="gbt", target_features=features_importance)
    df_importance = get_shap_importance(df_shap)
    plot_shap_importance(df_importance, color=j_purple, title="GBT feature importance")
end