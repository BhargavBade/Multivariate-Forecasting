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

        # Apply layer normalization
        out = self.layer_norm(output[:, -1, :])  # take the last output and normalize


        out = self.sigmoid(out)
        # out = self.relu(out)
        out = self.fc_1(out) # first dense
        out = self.dropout_fc(out)
        out = self.sigmoid(out) # relu
        # out = self.relu(out)
        out = self.fc_2(out) # final output
        
        return out
    
if __name__ == '__main__':
    
    # Example input parameters
    batch_size = 32
    sequence_length = 96
    input_features = 11
    hidden_size = 64
    output_size = 1
    num_layers = 3
    dropout = 0.2

    # Create example input tensor
    example_input = torch.rand(batch_size, sequence_length, input_features)
    
    # Instantiate the model
    model = LSTMModel(input_features, hidden_size, output_size, num_layers, dropout)

    # Forward pass
    output = model(example_input)
    
    # Print output shape and model structure for debugging
    print(f"Output shape: {output.shape}")
    print(model)

#----------------------------------------------------------------------------------------

# import torch
# from torch import nn

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # LSTM layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

#         # Fully connected layers with normalization
#         self.fc_1 = nn.Linear(hidden_size, 128)
#         self.bn1 = nn.BatchNorm1d(128)  # Batch normalization
#         self.fc_2 = nn.Linear(128, 64)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.fc_3 = nn.Linear(64, output_size)

#         # Activation and Dropout
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # LSTM forward pass
#         output, _ = self.lstm(x)  # No need to manually initialize h0, c0
#         out = output[:, -1, :]  # Take last time step output

#         # Fully connected layers with activations
#         out = self.fc_1(out)
#         out = self.bn1(out)  # Batch normalization helps training stability
#         out = self.relu(out)
#         out = self.dropout(out)

#         out = self.fc_2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.dropout(out)

#         out = self.fc_3(out)  # Final output layer (no activation for regression)

#         return out

# # Example usage
# if __name__ == '__main__':
#     batch_size = 32
#     sequence_length = 96
#     input_features = 11
#     hidden_size = 128  # Increased hidden size
#     output_size = 24  # Predicting 24 time steps ahead
#     num_layers = 3
#     dropout = 0.3

#     example_input = torch.rand(batch_size, sequence_length, input_features)
#     model = LSTMModel(input_features, hidden_size, output_size, num_layers, dropout)
#     output = model(example_input)

#     print(f"Output shape: {output.shape}")
#     print(model)



