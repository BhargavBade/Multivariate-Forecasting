
#DATA CONFIGS
batch_size = 64
seq_len = 192 #Lenght of the window
pred_len = 1 #No of steps to predict
label_len = 72 #No of target lookbacks

# Hyper parmas
lr = 0.0001
num_epochs = 100

# Features to feed into the network
enc_inp = 11 #No of features as Input to encoder
dec_in = 1 #No of features as Input to decoder
c_out = 1 #Nof of features to predict

wandb_name = "Informer_Model"
wandb_project = "Multivar_TS_Informer_Forecasting"
