
#DATA CONFIGS

#window_size = 120
n_steps_in = 48
n_steps_out = 1
#stride = 1
batch_size = 64


#LSTM NETWORK
input_size = 11  # number of features
hidden_size = 64 # number of hidden units in LSTM
output_size = 1  # output size
num_layers = 7   # number of LSTM layers
lr = 0.001
num_epochs = 200