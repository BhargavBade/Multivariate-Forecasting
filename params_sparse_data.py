
#DATA CONFIGS
batch_size = 24
seq_len = 182 #Lenght of the window
pred_len = 91 #No of steps to predict
label_len = 92 #No of target lookbacks #the total should be 365

nan_fraction = 0.5 #Fraction of NaNs in the data

# Hyper parmas
lr = 0.0001
num_epochs = 25

# Number of epochs to wait before stopping the training if no improvement in the val loss
early_stop_patience = 15
# Number of training epochs to skip for saving the model
warmup_epochs = 15  

# Features to feed into the network
total_feat = 14 #["year", "month", "day", "hour", "season", "cbwd", 'PM',  
                 # "precipitation", "DEWP", 'HUMI', 'PRES', 'TEMP', "Iprec", "Iws" ]
                 
# Selected Features to drop from the total features
no_of_drop_feat = 4
drop_feat = ["cbwd", "Iws", "precipitation", "Iprec"]

# Features reamining after dropping some features from the data
data_feat = total_feat - no_of_drop_feat #No of features used from the total_feat 

# Filtering features to send into the network
enc_inp = data_feat - 4 #No of features as Input to encoder (-4 due to removal of timestamp cols)
dec_in = data_feat - 4 #No of features as Input to decoder
c_out = data_feat - 4 #No of features to predict

## Define custom months to send into encoder, decoder and prediction for train, val and test sets
encoder_months = [10, 11, 12, 1, 2, 3]
decoder_months = [7, 8, 9]
pred_months = [4, 5, 6]

# Testing Params
num_samples_to_plot = 3 #random selection of samples out of available 24 samples
#Select which features to plot out of available features
feat_for_plotting = ['PM', 'HUMI', 'PRES', 'TEMP']

# WandB project details
wandb_name = "Informer_Model"
wandb_project = "Multivar_TS_Informer_Forecasting"
