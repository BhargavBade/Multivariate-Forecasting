#DATA CONFIGS
n_steps_in = 196
n_steps_out = 48
batch_size = 64

#LSTM NETWORK
input_size = 11  # number of features
hidden_size = 64 # number of hidden units in LSTM
output_size = 48  # output size
num_layers = 3   # number of LSTM layers
lr = 0.001
num_epochs = 300

wandb_name = "LSTM_Model"
wandb_project = "Multivar_TS_LSTM_Forecasting"