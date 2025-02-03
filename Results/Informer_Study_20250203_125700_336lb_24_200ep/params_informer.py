
#DATA CONFIGS
batch_size = 32
seq_len = 336 #Lenght of the window
pred_len = 24 #No of steps to predict
label_len = 336 #No of target lookbacks

# Hyper parmas
lr = 0.0001
num_epochs = 200

# Features to feed into the network
enc_inp = 11 #No of features as Input to encoder
dec_in = 1 #No of features as Input to decoder
c_out = 1 #No of features to predict

wandb_name = "Informer_Model"
wandb_project = "Multivar_TS_Informer_Forecasting"
