import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_1 = nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, output_size) # fully connected last layer
        self.relu = nn.ReLU()

        # Fully connected layer (for output)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))

        # Use the output of the last time step (not all layers or flattened output)
        out = output[:, -1, :]  # take the last output of the sequence

        #hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        #out = self.relu(hn)

        out = self.relu(out)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out

