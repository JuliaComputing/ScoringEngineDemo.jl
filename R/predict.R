# source("fit_model.R")  # Load your code.
source("load_model.R")
source("preprocess_X_data.R")
source("predict_expected_claim.R")
source("predict_premium.R")

# This script expects sys.args arguments for (1) the dataset and (2) the output file.
output_dir = Sys.getenv('OUTPUTS_DIR', '.')
input_dataset = Sys.getenv('DATASET_PATH', 'training_data.csv')  # The default value.
output_claims_file = paste(output_dir, 'claims.csv', sep = '/')  # The file where the expected claims should be saved.
output_prices_file = paste(output_dir, 'prices.csv', sep = '/')  # The file where the prices should be saved.
model_output_path = 'trained_model.RData'

args = commandArgs(trailingOnly=TRUE)

if(length(args) >= 1){
  input_dataset = args[1]
}
if(length(args) >= 2){
  output_claims_file = args[2]
}
if(length(args) >= 3){
  output_prices_file = args[3]
}

# Load the dataset.
# Remove the claim_amount column if it is in the dataset.
Xraw = read.csv(input_dataset)

if('claim_amount' %in% colnames(input_dataset)){
  Xraw = within(Xraw, rm('claim_amount'))
}

# Load the saved model, and run it.
trained_model = load_model(model_output_path)

if(Sys.getenv('WEEKLY_EVALUATION', 'false') == 'true') {
  prices <- predict_premium(trained_model, Xraw)
  write.table(x = prices, file = output_prices_file, row.names = FALSE, col.names=FALSE, sep = ",")

} else {
  claims <- predict_expected_claim(trained_model, Xraw)
  write.table(x = claims, file = output_claims_file, row.names = FALSE, col.names=FALSE, sep = ",")
}
