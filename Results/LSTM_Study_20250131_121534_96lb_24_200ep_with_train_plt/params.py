#DATA CONFIGS
n_steps_in = 96
n_steps_out = 24
batch_size = 32

#LSTM NETWORK
input_size = 11  # number of features
hidden_size = 64 # number of hidden units in LSTM
output_size = 24  # output size
num_layers = 4   # number of LSTM layers
lr = 0.001
num_epochs = 200

wandb_name = "LSTM_Model"
wandb_project = "Multivar_TS_LSTM_Forecasting"