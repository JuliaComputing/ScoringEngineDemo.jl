fit_model_freq_sev <- function (x_raw, y_raw){

  require("data.table")
  require("xgboost")
  # Model training function: given training data (X_raw, y_raw), train this pricing model.

  # Parameters
  # ----------
  # X_raw : Dataframe, with the columns described in the data dictionary.
  # 	Each row is a different contract. This data has not been processed.
  # y_raw : a array, with the value of the claims, in the same order as contracts in X_raw.
  # 	A one dimensional array, with values either 0 (most entries) or >0.

  # Returns
  # -------
  x_tot <- preprocess_X_data(x_raw)
  y_tot <- y_raw[["claim_amount"]]
  # y_tot[y_tot > 0] <- pmin(25000, y_tot[y_tot > 0])

  f_tot <- as.integer(y_tot > 0)

  xs_tot <- x_tot[y_tot > 0, ]
  s_tot <- y_tot[y_tot > 0]
  # s_tot <- pmin(25000, s_tot)

  models_f <- list()
  print("freq models")
  for (i in 1:5) {
    set.seed(123+i)
    train_id <- sample(1:nrow(x_tot), size = as.integer(0.9 * nrow(x_tot)), replace = F)
    # var_delete <- grep("^vh_mod_mat", colnames(x_tot))
    xgb_train <- xgb.DMatrix(data = x_tot[train_id, ], label = f_tot[train_id])
    xgb_eval <- xgb.DMatrix(data = x_tot[-train_id, ], label = f_tot[-train_id])

    params <- list(max_depth = 5, eta = 0.02, subsample = 0.9, colsample_bytree = 0.9, min_child_weight = 5, lambda = 1, alpha = 1, gamma=0.0,
                   tree_method = "hist", objective = "count:poisson", max_bin=32)

    model <- xgb.train(data = xgb_train, watchlist = list(eval=xgb_eval),
                       params = params,
                       nrounds = 800, verbose = 1,
                       print_every_n = 10L,
                       early_stopping_rounds = NULL)
    # xgboost::xgb.importance(model = model)
    models_f[[i]] <- model
  }


  models_s <- list()
  print("sev models")
  for (i in 1:5) {
    set.seed(123+i)
    train_id <- sample(1:nrow(xs_tot), size = as.integer(0.9 * nrow(xs_tot)), replace = F)
    xgb_train <- xgb.DMatrix(data = xs_tot[train_id, ], label = s_tot[train_id])
    xgb_eval <- xgb.DMatrix(data = xs_tot[-train_id, ], label = s_tot[-train_id])

    params <- list(max_depth = 4, eta = 0.02, subsample = 0.9, colsample_bytree = 0.9, min_child_weight = 5, lambda = 1, alpha = 1, gamma=0.0,
                   tree_method = "hist", objective = "reg:gamma", max_bin=16)

    model <- xgb.train(data = xgb_train, watchlist = list(eval=xgb_eval),
                       params = params,
                       nrounds = 600, verbose = 1,
                       print_every_n = 10L,
                       early_stopping_rounds = NULL)
    # xgboost::xgb.importance(model = model)
    models_s[[i]] <- model
  }


  models_t <- list()
  # print("tweedie models")
  # for (i in 1:5) {
  #   set.seed(123+i)
  #   train_id <- sample(1:nrow(x_tot), size = as.integer(0.9 * nrow(x_tot)), replace = F)
  #   xgb_train <- xgb.DMatrix(data = x_tot[train_id, ], label = y_tot[train_id])
  #   xgb_eval <- xgb.DMatrix(data = x_tot[-train_id, ], label = y_tot[-train_id])
  #
  #   params <- list(max_depth = 5, eta = 0.02, subsample = 0.9, colsample_bytree = 0.9, min_child_weight = 16, lambda = 8, alpha = 1, gamma=0.1,
  #                  tree_method = "hist", objective = "reg:tweedie", eval_metric = "rmse", max_bin=32)
  #
  #   model <- xgb.train(data = xgb_train, watchlist = list(eval=xgb_eval),
  #                      params = params,
  #                      nrounds = 500, verbose = 1,
  #                      print_every_n = 10L,
  #                      early_stopping_rounds = NULL)
  #   # xgboost::xgb.importance(model = model)
  #   models_t[[i]] <- model
  # }

  models <- list(models_f=models_f, models_s=models_s, models_t=models_t)

  return(models)  # return(trained_model)
}
