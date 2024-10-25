#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import torch
import numpy as np
import shutil


# In[11]:


import importlib
import sys
import matplotlib.pyplot as plt

sys.path.append('.')  # Add the parent directory to the Python path
import params
importlib.reload(params)


# In[12]:


print(params.window_size)


# In[13]:


print(params.hidden_size)


# In[14]:


#Study Folder creation for saving Results and Plots
sys.path.append('./Others') 
import study_folder
importlib.reload(study_folder)
study_folder, train_folder, test_folder, val_folder = study_folder.create_study_folder()
print(study_folder)
print(train_folder)


# In[15]:


#saving the params file used for this study in this study folder

# Path to the config.py file in your project directory
config_file_path = './params.py'

# Destination where the config.py file will be copied (inside the study folder)
config_destination_path = os.path.join(study_folder, 'params.py')

# Copy the config.py file to the study folder
shutil.copy(config_file_path, config_destination_path)

print(f'Config file saved to: {config_destination_path}')


# In[16]:


import sys
sys.path.append('./Data')  # Add the current directory to Python path
import prepare_data
importlib.reload(prepare_data)
from prepare_data import DataPreparer


# In[17]:


# Create an instance of DataPreparer
data_preparer = DataPreparer(data_dir='./01_PM2.5 Chinese Weather data')


# In[18]:


# Prepare the data (loads, cleans, splits, and creates tensors)
data_preparer.prepare_data()

# Get the tensors
train_data_tensor, train_labels_tensor, val_data_tensor, val_labels_tensor, test_data_tensor, test_labels_tensor, scaler, pm_index = data_preparer.get_tensors()


print('PM index during scaling is:',pm_index)
# Now you can use these tensors for training in your notebook
print("Train data tensor shape:", train_data_tensor.shape)
print("Train labels tensor shape:", train_labels_tensor.shape)

print("Val data tensor shape:", val_data_tensor.shape)
print("Val labels tensor shape:", val_labels_tensor.shape)

print("Test data tensor shape:", test_data_tensor.shape)
print("Test labels tensor shape:", test_labels_tensor.shape)


# In[19]:


import torch.nn as nn
import torch.optim as optim


# In[20]:


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected layer (for output)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output from all time steps through the fully connected layer
        out = self.fc(out)  # out shape will now be [batch_size, sequence_length, output_size]
        return out


# In[21]:


# Prepare the dataset and dataloader
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data.to(torch.float32)
        self.labels = labels.to(torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Data and Labels
X_train = train_data_tensor.to(torch.float32)
y_train = train_labels_tensor.to(torch.float32)

# Dataset and Dataloader
train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params.batch_size, 
                                           shuffle = False, drop_last=True)


# In[22]:


# Define model, loss function, and optimizer
input_size = params.input_size  # number of features
hidden_size = params.hidden_size  # number of hidden units in LSTM
output_size = params.output_size  # output size (1 for regression, could be different for classification)
num_layers = params.num_layers  # number of LSTM layers

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.MSELoss()  # For regression, use nn.CrossEntropyLoss() for classification
optimizer = optim.Adam(model.parameters(), lr = params.lr)
#optimizer = optim.Adam(model.parameters(), lr = 0.001)


# In[23]:


## Randomly select 12 sequences from the training set for plotting
#num_fixed_sequences = 12
#random_indices = torch.randperm(len(X_train))[:num_fixed_sequences]  # Get 12 random indices

## Select the corresponding sequences and labels
#fixed_sequences = X_train[random_indices].to('cuda' if torch.cuda.is_available() else 'cpu')
#fixed_labels = y_train[random_indices].to('cuda' if torch.cuda.is_available() else 'cpu')


# In[24]:


# First 12 sequences in the train data set
num_fixed_sequences = 12

# Select the first 12 sequences and labels
fixed_sequences = X_train[:num_fixed_sequences].to('cuda' if torch.cuda.is_available() else 'cpu')
fixed_labels = y_train[:num_fixed_sequences].to('cuda' if torch.cuda.is_available() else 'cpu')


# In[25]:


#Train the model
num_epochs = params.num_epochs
#num_epochs = 10

# Initialize list to store loss values
loss_values = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data, labels in train_loader:
        data, labels = data.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)

        # Compute the loss (squeeze the outputs so they match the shape of labels)
        loss = criterion(outputs.squeeze(-1), labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    loss_values.append(avg_loss)  # Store average loss for this epoch

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Every 10 epochs, generate plots for the fixed sequences
    if (epoch + 1) % 20 == 0:
        model.eval()  # Switch to evaluation mode for generating predictions
        with torch.no_grad():
            predictions = model(fixed_sequences)

        # Mean and standard deviation of the scaled values
        means = scaler.mean_
        stds = scaler.scale_
        
        # Reverse the standardization to get predictions and actual values in the original scale
        predictions_np = predictions.squeeze(-1).cpu().numpy() * stds[pm_index] + means[pm_index] # Reverse standardization
        #predictions_np = predictions.squeeze(-1).cpu().numpy()
        actual_np = fixed_labels.cpu().numpy() * stds[pm_index] + means[pm_index]
       # actual_np = fixed_labels.cpu().numpy()

         # Create two images, each with 6 subplots
        fig1, axs1 = plt.subplots(2, 3, figsize=(15, 10))  # First image, 2 rows x 3 columns
        fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))  # Second image, 2 rows x 3 columns
    
        for i in range(6):  # First 6 sequences in the first figure
            ax = axs1[i // 3, i % 3]  # Access subplot in grid
            ax.plot(predictions_np[i], label=f'Predicted Sequence {i+1}')
            ax.plot(actual_np[i], label=f'Actual Sequence {i+1}', linestyle='dashed')
            ax.set_title(f'Sequence {i+1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('PM2.5')
            ax.legend()
    
        for i in range(6, 12):  # Next 6 sequences in the second figure
            ax = axs2[(i - 6) // 3, (i - 6) % 3]  # Access subplot in grid
            ax.plot(predictions_np[i], label=f'Predicted Sequence {i+1}')
            ax.plot(actual_np[i], label=f'Actual Sequence {i+1}', linestyle='dashed')
            ax.set_title(f'Sequence {i+1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('PM2.5')
            ax.legend()
    
        # Display the two images with 6 subplots each
        # Save the figures in the study folder
        fig1_path = os.path.join(train_folder, f'Epoch_{epoch + 1}_Sequences_1-6.png')
        fig2_path = os.path.join(train_folder, f'Epoch_{epoch + 1}_Sequences_7-12.png')
        fig1.suptitle(f'Epoch {epoch + 1}: Actual vs. Predicted PM2.5 (Sequences 1-6)')
        fig2.suptitle(f'Epoch {epoch + 1}: Actual vs. Predicted PM2.5 (Sequences 7-12)')

        fig1.savefig(fig1_path)  # Save the first figure
        fig2.savefig(fig2_path)  # Save the second figure

        plt.show()  # Show the first figure
        plt.show()  # Show the second figure

        plt.close(fig1)  # Close the figure to free memory
        plt.close(fig2)  # Close the figure to free memory

        print(f"Saved training plots for Epoch {epoch + 1} in {train_folder}")
    
        model.train()  # Switch back to training mode


# In[26]:


# Plot the loss curve after training
lossfig_path = os.path.join(train_folder, f'loss_curve.png')
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig(lossfig_path) 
plt.show()
plt.close()


# In[27]:


val_dataset = TimeSeriesDataset(val_data_tensor, val_labels_tensor)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size = params.batch_size, shuffle=False)


# In[28]:


# Set the model to evaluation mode
model.eval()

# Initialize a list to store predictions
all_predictions = []
all_actual_labels = []

# Initialize total loss for calculating MSE over the validation set
total_val_loss = 0.0

# Path to save the validation results
val_results_file = os.path.join(val_folder, 'validation_results.txt')

# Make predictions on the validation dataset using the DataLoader
with torch.no_grad():
    for data, labels in val_loader:
        #data_tensor, _ = data  # We only need the features for prediction
        #predictions = model(data_tensor.to('cuda' if torch.cuda.is_available() else 'cpu'))
        data, labels = data.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Forward pass to get predictions
        predictions = model(data)
        
        # Compute the validation loss (MSE)
        val_loss = criterion(predictions.squeeze(-1), labels)
        
        # Accumulate the total validation loss
        total_val_loss += val_loss.item()
        
        # Store predictions and actual labels for later analysis or metrics calculation
        all_predictions.append(predictions.cpu().numpy())  # Move predictions to CPU and store
        all_actual_labels.append(labels.cpu().numpy())     # Move labels to CPU and store

# Calculate the average MSE over the entire validation set
avg_val_loss = total_val_loss / len(val_loader)
print(f'Validation Loss (MSE): {avg_val_loss:.4f}')

# Concatenate all predictions and actual labels into a single array
val_predictions = np.concatenate(all_predictions, axis=0)
val_actual_labels = np.concatenate(all_actual_labels, axis=0)

# Write the validation loss to a text file
with open(val_results_file, 'w') as f:
    f.write(f'Validation Loss (MSE): {avg_val_loss:.4f}\n')
    f.write(f'Validation Predictions Shape: {val_predictions.shape}\n')
    f.write(f'Validation Actual Labels Shape: {val_actual_labels.shape}\n')

print(f'Validation results saved to: {val_results_file}')        
        
        
       # all_predictions.append(predictions.cpu().numpy())  # Append to list

## Concatenate all predictions into a single array
#val_predictions = np.concatenate(all_predictions, axis=0)


# In[29]:


# Step 2: Reverse Standardization for Predictions
means = scaler.mean_
stds = scaler.scale_

# Flatten predictions
predictions_np = val_predictions.squeeze(-1)
predictions_original_scale = predictions_np * stds[pm_index] + means[pm_index]  # Assuming PM2.5 is the first column
#predictions_original_scale = predictions_np 
## Step 3: Reverse Standardization for Actual PM2.5 Labels
#actual_labels_np = val_labels_tensor.numpy()
#actual_labels_original_scale = actual_labels_np * stds[pm_index] + means[pm_index]  # Same column assumption

# Step 3: Reverse Standardization for Actual PM2.5 Labels
actual_labels_original_scale = val_actual_labels * stds[pm_index] + means[pm_index]
#actual_labels_original_scale = val_actual_labels 



# In[30]:


import matplotlib.pyplot as plt
from IPython.display import display

# Number of sequences to plot
num_sequences = 10

# Loop through the first 10 sequences and create individual plots
for i in range(num_sequences):
    plt.figure(figsize=(14, 7))
    
    # Plot actual PM2.5 values
    plt.plot(actual_labels_original_scale[i], label='Actual PM2.5', color='blue', alpha=0.7)
    
    # Plot predicted PM2.5 values
    plt.plot(predictions_original_scale[i], label='Predicted PM2.5', color='red', alpha=0.7)
    
    # Set title and labels
    plt.title(f'Sequence {i+1}: Predicted vs Actual PM2.5 Values')
    plt.xlabel('Timestep')
    plt.ylabel('PM2.5 Value')
    
    # Show legend and grid
    plt.legend()
    plt.grid(True)
    
    # Display the plot inline
    display(plt.gcf())  # This will display the plot inline in notebooks

     # Save the plot to the 'val_folder' with an appropriate file name
    plot_filename = os.path.join(val_folder, f'sequence_{i+1}_val_plot.png')
    plt.savefig(plot_filename)  # Save figure
    
    # Close the current figure after displaying to avoid overlap
    plt.close()


# In[ ]:




