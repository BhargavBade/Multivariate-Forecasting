
#DATA CONFIGS
window_size = 96
stride = 1
batch_size = 32


#LSTM NETWORK
input_size = 15  # number of features
hidden_size = 32  # number of hidden units in LSTM
output_size = 1  # output size (1 for regression, could be different for classification)
num_layers = 3  # number of LSTM layers
lr = 0.0001
num_epochs = 200
