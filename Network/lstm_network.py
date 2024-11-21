import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Fully connected layers
        self.fc_1 = nn.Linear(hidden_size, 32) # fully connected 
        self.fc_2 = nn.Linear(32, output_size) # fully connected last layer
        
        # #Activation and Dropout
        # self.relu = nn.ReLU()
        # self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout_fc = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))

        # # Use the output of the last time step (not all layers or flattened output)
        # out = output[:, -1, :]  # take the last output of the sequence
        
        # Apply layer normalization
        out = self.layer_norm(output[:, -1, :])  # take the last output and normalize


        out = self.sigmoid(out)
        out = self.fc_1(out) # first dense
        out = self.dropout_fc(out)
        out = self.sigmoid(out) # relu
        out = self.fc_2(out) # final output
        
        return out

