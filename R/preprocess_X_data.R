preprocess_X_data <- function (x_raw){
  # Data preprocessing function: given X_raw, clean the data for training or prediction.

  require("data.table")
  require("xgboost")
  # x_raw <- Xdata

  # Parameters
  # ----------
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.

  # Returns
  # -------
  # A cleaned / preprocessed version of the dataset

  # YOUR CODE HERE ------------------------------------------------------
  dt <- data.table(x_raw)

  # dt[, c("id_policy") := NULL]
  dt[, density := population / town_surface_area]
  dt[, has_drv2 := ifelse(drv_sex2==0, 0, 1)]
  dt[, is_drv2_male := ifelse(drv_sex2=="M", 1, 0)]
  dt[, vh_type_commercial := ifelse(vh_type=="Commercial", 1, 0)]
  dt[, is_pol_payd := ifelse(pol_payd=="Yes", 1, 0)]
  dt[, drv_age1_male := ifelse(drv_sex1=="M", drv_age1, NA)]
  dt[, drv_age1_age2_diff := drv_age1 - drv_age2]
  dt[, vh_value_div_weight := vh_value / vh_weight]

  vars_num <- c("density", "year", "pol_no_claims_discount", "pol_duration", "pol_sit_duration",
                "drv_age1", "drv_age_lic1", "vh_age", "vh_speed", "vh_value", "vh_weight", "has_drv2", "is_drv2_male", "vh_type_commercial",
                "drv_age2", "drv_age_lic2", "is_pol_payd", "drv_age1_male", "drv_age1_age2_diff",
                "population", "town_surface_area", "vh_value_div_weight")

  dt[, (vars_num) := lapply(.SD, function(x) ifelse(is.na(x), median(x, na.rm=T), x)), .SDcols=vars_num]

  # ordered factors
  dt[, pol_coverage_fac := as.integer(factor(pol_coverage, levels = c("Min", "Med1", "Med2", "Max")))]
  dt[, pol_pay_freq_fac := as.integer(factor(pol_pay_freq, levels = c("Monthly", "Quarterly", "Biannual", "Yearly")))]
  dt[, drv_sex1_fac := as.integer(factor(drv_sex1, levels = c("F", "M")))]
  vars_fac <- c("pol_coverage_fac", "pol_pay_freq_fac", "drv_sex1_fac")

  # create matrix with numeric and factors
  vars_keep <- c(vars_num, vars_fac)
  x_mat <- as.matrix(dt[, ..vars_keep])

  # one-hot features
  dt[, pol_usage_mat := factor(pol_usage, levels = c("AllTrips", "Professional", "Retired", "WorkPrivate"))]
  pol_usage_mat = model.matrix(~ . - 1, dt[, "pol_usage_mat"])

  dt[, vh_fuel_mat := factor(vh_fuel, levels = c("Diesel", "Gasoline", "Hybrid"))]
  vh_fuel_mat = model.matrix(~ . - 1, dt[, "vh_fuel_mat"])

  dt[, pol_coverage_mat := factor(pol_coverage, levels = c("Min", "Med1", "Med2", "Max"))]
  pol_coverage_mat = model.matrix(~ . - 1, dt[, "pol_coverage_mat"])

  dt[, pol_pay_freq_mat := factor(pol_pay_freq, levels = c("Monthly", "Quarterly", "Biannual", "Yearly"))]
  pol_pay_freq_mat = model.matrix(~ . - 1, dt[, "pol_pay_freq_mat"])


  # veh make model
  dt[, vh_mod_mat := factor(vh_make_model, levels = c("zoypfizhpbtpjwpv", "xkzehzohmfrsmolg", "iulvirmzdntweaee", "kvcddisqpkysmvvo", "prtnwsypyfnshpqx", "aparvvfowrjncdhp", "xjaddkudsebowzen", "ponwkmeaxagundzq",
                                                      "zqruwnlzuefcpqjm", "kzwthrslljkmbqur", "iwhqpdfuhrsxyqxe", "jjycmklnkdivnypu", "svmjzfcsvgxiwwjt", "hselphnqlvecmmyx", "lqkdgbosdzrtitgx", "tdgkjlphosocwbgu",
                                                      "swjkmyqytzxjwgag", "biqzvbfzjivqmrro", "johsjccpkithubii", "rthsjeyjgdlmkygk", "OTHER")
  )]
  dt[is.na(vh_mod_mat), vh_mod_mat := "OTHER"]
  vh_mod_mat = model.matrix(~ . - 1, dt[, "vh_mod_mat"])

  # add one-hot features to matrix
  x_mat <- cbind(x_mat, pol_usage_mat, vh_fuel_mat, pol_coverage_mat, pol_pay_freq_mat, vh_mod_mat)
  # ---------------------------------------------------------------------
  # The result trained_model is something that you will save in the next section
  return(x_mat) # change this to return the cleaned data
}
