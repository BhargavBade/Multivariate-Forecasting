#!/usr/bin/env python
# coding: utf-8

# In[34]:

import os
import torch
import numpy as np
import shutil
import random
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from Network.lstm_network import LSTMModel
from matplotlib.ticker import MaxNLocator
# In[35]:

import importlib
import sys
import matplotlib.pyplot as plt

sys.path.append('.')  # Add the parent directory to the Python path
import params
importlib.reload(params)

# In[36]:

#print(params.window_size)
print(params.n_steps_in)


# In[37]:

print(params.hidden_size)


# In[38]:


#Study Folder creation for saving Results and Plots
sys.path.append('./Others') 
import study_folder
importlib.reload(study_folder)
study_folder, train_folder, test_folder, val_folder = study_folder.create_study_folder()
print(study_folder)
print(train_folder)


# In[39]:

#saving the params file used for this study in this study folder

# Path to the config.py file in your project directory
config_file_path = './params.py'

# Destination where the config.py file will be copied (inside the study folder)
config_destination_path = os.path.join(study_folder, 'params.py')

# Copy the config.py file to the study folder
shutil.copy(config_file_path, config_destination_path)

print(f'Config file saved to: {config_destination_path}')


# In[40]:


import sys
sys.path.append('./Data')  # Add the current directory to Python path
import prepare_data01
importlib.reload(prepare_data01)
from prepare_data01 import DataPreparer


# In[41]:


# Create an instance of DataPreparer
data_preparer = DataPreparer(data_dir='./01_PM2.5 Chinese Weather data')


# In[42]:

# Prepare the data (loads, cleans, splits, and creates tensors)

data_preparer.prepare_data()

# Get the tensors
(train_data_tensor, train_labels_tensor, val_data_tensor, val_labels_tensor, past_pm25_val, val_pm_dt,
 test_data_tensor, test_labels_tensor, past_pm25_test, test_pm_dt, scaler, pm_index) = data_preparer.get_tensors()


print('PM index during scaling is:',pm_index)
# Now you can use these tensors for training in your notebook
print("Train data tensor shape:", train_data_tensor.shape)
print("Train labels tensor shape:", train_labels_tensor.shape)

print("Val data tensor shape:", val_data_tensor.shape)
print("Val labels tensor shape:", val_labels_tensor.shape)

print("Test data tensor shape:", test_data_tensor.shape)
print("Test labels tensor shape:", test_labels_tensor.shape)

# In[45]:

# Prepare the datasets and dataloaders

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data.to(torch.float32)
        self.labels = labels.to(torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Train Data and Train Labels
X_train = train_data_tensor.to(torch.float32)
y_train = train_labels_tensor.to(torch.float32)

# Train Dataset and Train Dataloader
train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params.batch_size, 
                                           shuffle = False, drop_last=True)

# Val Data and Val Labels
X_val = val_data_tensor.to(torch.float32)
y_val = val_labels_tensor.to(torch.float32)

# Val Dataset and Val Dataloader
val_dataset = TimeSeriesDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size = params.batch_size, 
                                         shuffle=False, drop_last=True)

# Test Data and Test Labels
X_test = test_data_tensor.to(torch.float32)
y_test = test_labels_tensor.to(torch.float32)

# Test Dataset and Test Dataloader
test_dataset = TimeSeriesDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params.batch_size, 
                                          shuffle=False, drop_last=True)


# In[46]:

# Define model, loss function, and optimizer

input_size = params.input_size  # number of features
hidden_size = params.hidden_size  # number of hidden units in LSTM
output_size = params.output_size  # output size (1 for regression, could be different for classification)
num_layers = params.num_layers  # number of LSTM layers

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
#criterion = nn.MSELoss()  # For regression, use nn.CrossEntropyLoss() for classification
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = params.lr)

# In[47]:

# Randomly select 12 sequences from the training set for plotting
num_fixed_sequences = 12
random_indices = torch.randperm(len(X_train))[:num_fixed_sequences]  # Get 12 random indices

# Select the corresponding sequences and labels
fixed_sequences = X_train[random_indices].to('cuda' if torch.cuda.is_available() else 'cpu')
fixed_labels = y_train[random_indices].to('cuda' if torch.cuda.is_available() else 'cpu')


# In[48]:


## First 12 sequences in the train data set
#num_fixed_sequences = 6

## Select the first 12 sequences and labels
#fixed_sequences = X_train[:num_fixed_sequences].to('cuda' if torch.cuda.is_available() else 'cpu')
#fixed_labels = y_train[:num_fixed_sequences].to('cuda' if torch.cuda.is_available() else 'cpu')


# In[49]:

#Storing best model based on test loss    
best_test_loss = float('inf')  # Initialize best test loss as infinity
checkpoint_path = os.path.join(train_folder, 'best_model.pth')  # Path to save the best model    
    
####################################################################################################

#Train the model

num_epochs = params.num_epochs

# Initialize list to store loss values
loss_values = []
test_losses = []   # Store test loss for each 5th epoch

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
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    loss_values.append(avg_loss)  # Store average loss for this epoch

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # -------- Testing Phase (every 5 epochs) --------
    if (epoch + 1) % 5 == 0:
        model.eval()  # Set the model to evaluation mode
        running_test_loss = 0.0

        with torch.no_grad():  # Disable gradient calculations for testing
            for data, labels in test_loader:
                data, labels = data.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                # Forward pass
                outputs = model(data)

                # Compute the loss
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()

        # Compute average test loss for the epoch
        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Print test loss for the current evaluation
        print(f'-- Evaluation after Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}')
        
        # Check if this is the best test loss so far
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            print(f'New best test loss: {best_test_loss:.4f}. Saving model.')
            torch.save(model.state_dict(), checkpoint_path)  # Save model
        
    # Every 10 epochs, generate plots for the fixed sequences
    if (epoch + 1) % 20 == 0:
        model.eval()  # Switch to evaluation mode for generating predictions
        with torch.no_grad():
            predictions = model(fixed_sequences)

        # Mean and standard deviation of the scaled values
        means = scaler.mean_
        stds = scaler.scale_
        
        # Reverse the standardization to get pm predictions and actual values in the original scale
        predictions_np = predictions.squeeze(-1).cpu().numpy() * stds[pm_index] + means[pm_index] 
        #predictions_np = predictions.squeeze(-1).cpu().numpy()
        actual_np = fixed_labels.cpu().numpy() * stds[pm_index] + means[pm_index]
        # actual_np = fixed_labels.cpu().numpy()

          # Create two images, each with 6 subplots
        fig1, axs1 = plt.subplots(2, 3, figsize=(15, 10))  # First image, 2 rows x 3 columns
        #fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))  # Second image, 2 rows x 3 columns
    
        for i in range(6):  # First 6 sequences in the first figure
            ax = axs1[i // 3, i % 3]  # Access subplot in grid
            ax.plot(predictions_np[i], label=f'Predicted Sequence {i+1}')
            ax.plot(actual_np[i], label=f'Actual Sequence {i+1}', linestyle='dashed')
            ax.set_title(f'Sequence {i+1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('PM2.5')
            ax.legend()
    
        # Display the two images with 6 subplots each
        # Save the figures in the study folder
        fig1_path = os.path.join(train_folder, f'Epoch_{epoch + 1}_Sequences_1-6.png')
        fig1.suptitle(f'Epoch {epoch + 1}: Actual vs. Predicted PM2.5 (Sequences 1-6)')

        fig1.savefig(fig1_path)  # Save the first figure

        plt.show()  # Show the first figure
        plt.close(fig1)  # Close the figure to free memory
        print(f"Saved training plots for Epoch {epoch + 1} in {train_folder}")
        
        model.train()  # Switch back to training mode


# In[50]:

# Plot the loss curve after training
lossfig_path = os.path.join(train_folder, 'loss_curve.png')
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(lossfig_path) 
plt.show()
plt.close()

##############################################################################################
# In[52]:
    
# Valdiation Set Evaluation

# Load the best model after training
print(f"Loading the best model from checkpoint with test loss: {best_test_loss:.4f}")
model.load_state_dict(torch.load(checkpoint_path, weights_only = True))

# Set the model to evaluation mode
model.eval()

# Initialize a list to store predictions
val_predictions = []
val_actual_labels = []

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
        #val_loss = criterion(predictions.squeeze(-1), labels)
        val_loss = criterion(predictions, labels)
        
        # Accumulate the total validation loss
        total_val_loss += val_loss.item()
        
        # Store predictions and actual labels for later analysis or metrics calculation
        val_predictions.append(predictions.cpu().numpy())  # Move predictions to CPU and store
        val_actual_labels.append(labels.cpu().numpy())     # Move labels to CPU and store

# Calculate the average MSE over the entire validation set
avg_val_loss = total_val_loss / len(val_loader)
print(f'Validation Loss (MSE): {avg_val_loss:.4f}')

# Concatenate all predictions and actual labels into a single array
val_predictions = np.concatenate(val_predictions, axis=0)
val_actual_labels = np.concatenate(val_actual_labels, axis=0)

# Write the validation loss to a text file
with open(val_results_file, 'w') as f:
    f.write(f'Validation Loss (MSE): {avg_val_loss:.4f}\n')
    f.write(f'Validation Predictions Shape: {val_predictions.shape}\n')
    f.write(f'Validation Actual Labels Shape: {val_actual_labels.shape}\n')

print(f'Validation results saved to: {val_results_file}')        
        
# In[53]:

# Step 2: Reverse Standardization for single step Validation set Predictions
means = scaler.mean_
stds = scaler.scale_

# Flatten predictions
#predictions_np = val_predictions.squeeze(-1)
val_predictions_np = val_predictions
val_predictions_original_scale = val_predictions_np * stds[pm_index] + means[pm_index]  # Assuming PM2.5 is the first column

# Step 3: Reverse Standardization for Actual PM2.5 Labels
val_actual_labels_original_scale = val_actual_labels * stds[pm_index] + means[pm_index]

# Step 4: Reverse Standardization for Past PM2.5 val data
past_pm25_val_org_scl = past_pm25_val * stds[pm_index] + means[pm_index]

######################################################################################

# #Revere scaling for multistep predictor

# val_predictions_original_scale = (val_predictions_np * stds[pm_index] + means[pm_index]).reshape(-1, 4)  # Assuming PM2.5 is the first column
# #predictions_original_scale = predictions_np 

# # Step 3: Reverse Standardization for Actual PM2.5 Labels
# val_actual_labels_original_scale = (val_actual_labels * stds[pm_index] + means[pm_index]).reshape(-1, 4)
# #actual_labels_original_scale = val_actual_labels 



#########################################################################################

# Extract mean and std for the first four columns (year, month, day, hour)
mean_dt = scaler.mean_[:4]    # Mean of the first 4 columns
std_dt = scaler.scale_[:4]     # Standard deviation of the first 4 columns

# Reverse the scaling for date-time columns in val_pm_dt
val_pm_dt_original = (val_pm_dt * std_dt) + mean_dt

# (Optional) Convert to integers if needed for exact date-time representation
val_pm_dt_original = np.round(val_pm_dt_original)


# In[ ]:

# # Number of sequences to plot
# num_sequences = 10
# past_window_size = params.n_steps_in
# future_window_size = params.n_steps_out

# # Loop through the first 10 sequences and create individual plots
# for i in range(num_sequences):
#     plt.figure(figsize=(14, 7))
    
    
#     plt.plot(range(past_window_size), past_pm25_val_org_scl[i], label='PM2.5 (Past Data)', color='blue', alpha=0.7)
    
#     #Plot actual PM2.5 values
#     #plt.plot(actual_labels_original_scale[i], label='Actual PM2.5', color='blue', alpha=0.7)
#     plt.plot(range(past_window_size, past_window_size + future_window_size), 
#              val_actual_labels_original_scale[i], label='Target (Future Data)', color='orange', alpha=0.7)
    
#     # Plot predicted PM2.5 values
#     #plt.plot(predictions_original_scale[i], label='Predicted PM2.5', color='red', alpha=0.7)
#     plt.plot(range(past_window_size, past_window_size + future_window_size), 
#             val_predictions_original_scale[i], label='Predicted (Future Data)', color='green', alpha=0.7)
    
    
#     # Set title and labels
#     plt.title(f'Sequence {i+1}: Predicted vs Actual PM2.5 Values')
#     plt.xlabel('Timestep')
#     plt.ylabel('PM2.5 Value')
    
#     # Show legend and grid
#     plt.legend()
#     plt.grid(True)
    
#     # Display the plot inline
#     display(plt.gcf())  # This will display the plot inline in notebooks

#      # Save the plot to the 'val_folder' with an appropriate file name
#     plot_filename = os.path.join(val_folder, f'sequence_{i+1}_val_plot.png')
#     plt.savefig(plot_filename)  # Save figure
    
#     # Close the current figure after displaying to avoid overlap
#     plt.close()
##########################################################################################
##########################################################################################
##########################################################################################

# # Apply the inverse scaling function and get date-time columns for validation data
# val_data_original_scale, val_dates = inverse_scale_with_reconstruction(
#     val_data_tensor, past_pm25_val, scaler, pm_index, date_indices=[0, 1, 2, 3])

##########################################################################################
##########################################################################################

#Plotting All VAL predictions at a time with dates


# Convert year, month, day, hour into datetime format strings for plotting
timestamps = [datetime(int(year), int(month), int(day), int(hour)).strftime('%Y-%m-%d %H')
    for year, month, day, hour in val_pm_dt_original[:, 0, :]]

# Plotting the Validation Predictions vs Actual PM2.5 Values with Date-Time Labels
plt.figure(figsize=(14, 7))
plt.plot(val_actual_labels_original_scale, label='Val Actual PM2.5', color='blue', alpha=0.7)
plt.plot(val_predictions_original_scale, label='Val Predicted PM2.5', color='orange', alpha=0.7)
plt.title('Val Predicted vs Val Actual PM2.5 Values')
plt.xlabel('Date-Time')
plt.ylabel('PM2.5 Value')
plt.legend()
plt.grid(True)

# Set x-axis ticks to display every nth timestamp to avoid overlap
n = 50  # Adjust n to control the frequency of x-axis labels for readability
plt.xticks(range(0, len(timestamps), n), timestamps[::n], rotation=45, ha='right')


# Force Matplotlib to only show the specified ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# Save the plot to the 'val_folder' with an appropriate file name
plot_filename = os.path.join(val_folder, 'val_pred_plot.png')
plt.savefig(plot_filename, dpi=600)  # Save figure

plt.show()

# Close the current figure after displaying to avoid overlap
plt.close()



###################################################################################################
###################################################################################################
###################################################################################################

#TESTING

#Evaluation of the TESTING Set

# Set the model to evaluation mode
model.eval()

# Initialize a list to store predictions
test_predictions = []
test_actual_labels = []

# Initialize total loss for calculating MSE over the validation set
total_test_loss = 0.0

# Path to save the validation results
test_results_file = os.path.join(test_folder, 'test_results.txt')

# Make predictions on the validation dataset using the DataLoader
with torch.no_grad():
    for data, labels in test_loader:
        #data_tensor, _ = data  # We only need the features for prediction
        #predictions = model(data_tensor.to('cuda' if torch.cuda.is_available() else 'cpu'))
        data, labels = data.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Forward pass to get predictions
        predictions = model(data)
        
        # Compute the validation loss (MSE)
        #val_loss = criterion(predictions.squeeze(-1), labels)
        test_loss = criterion(predictions, labels)
        
        # Accumulate the total validation loss
        total_test_loss += test_loss.item()
        
        # Store predictions and actual labels for later analysis or metrics calculation
        test_predictions.append(predictions.cpu().numpy())  # Move predictions to CPU and store
        test_actual_labels.append(labels.cpu().numpy())     # Move labels to CPU and store

# Calculate the average MSE over the entire validation set
avg_test_loss = total_test_loss / len(test_loader)
print(f'Avg Test Loss (MSE): {avg_test_loss:.4f}')

# Concatenate all predictions and actual labels into a single array
test_predictions = np.concatenate(test_predictions, axis=0)
test_actual_labels = np.concatenate(test_actual_labels, axis=0)

# Write the validation loss to a text file
with open(test_results_file, 'w') as f:
    f.write(f'Test Loss (MSE): {avg_val_loss:.4f}\n')
    f.write(f'Test Predictions Shape: {val_predictions.shape}\n')
    f.write(f'Test Actual Labels Shape: {val_actual_labels.shape}\n')

print(f'Test results saved to: {test_results_file}')        
        
# In[53]:

# Step 2: Reverse Standardization for single step test set Predictions
means = scaler.mean_
stds = scaler.scale_

# Flatten predictions
#predictions_np = val_predictions.squeeze(-1)
test_predictions_np = test_predictions
test_predictions_original_scale = test_predictions_np * stds[pm_index] + means[pm_index]  # Assuming PM2.5 is the first column
#predictions_original_scale = predictions_np 

# Step 3: Reverse Standardization for Actual PM2.5 Labels
test_actual_labels_original_scale = test_actual_labels * stds[pm_index] + means[pm_index]

# Step 4: Reverse Standardization for Past PM2.5 val data
past_pm25_test_org_scl = past_pm25_test * stds[pm_index] + means[pm_index]

#################################################################################################

# Extract mean and std for the first four columns (year, month, day, hour)
mean_dt = scaler.mean_[:4]    # Mean of the first 4 columns
std_dt = scaler.scale_[:4]     # Standard deviation of the first 4 columns

# Reverse the scaling for date-time columns in val_pm_dt
test_pm_dt_original = (test_pm_dt * std_dt) + mean_dt

# (Optional) Convert to integers if needed for exact date-time representation
test_pm_dt_original = np.round(test_pm_dt_original)

#################################################################################################
#################################################################################################
##################################################################################################

# #Revere scaling for multistep predictor

# test_predictions_original_scale = (test_predictions_np * stds[pm_index] + means[pm_index]).reshape(-1, 4)  # Assuming PM2.5 is the first column
# #predictions_original_scale = predictions_np 

# # Step 3: Reverse Standardization for Actual PM2.5 Labels
# test_actual_labels_original_scale = (test_actual_labels * stds[pm_index] + means[pm_index]).reshape(-1, 4)
# #actual_labels_original_scale = val_actual_labels 



# In[ ]:
# Test set plotting with past PM2.5 and predicted PM2.5

#from IPython.display import display

# # Number of sequences to plot
# num_sequences = 20
# past_window_size = params.n_steps_in
# future_window_size = params.n_steps_out

# # Get total number of samples in the test set
# total_samples = len(test_predictions_original_scale)

# # Randomly select `num_sequences` unique indices from the test set
# random_indices = np.random.choice(total_samples, size=num_sequences, replace=False)

# # Loop through the first 10 sequences and create individual plots
# #for i in range(num_sequences):
# for i, idx in enumerate(random_indices):    
#     plt.figure(figsize=(14, 7))
    
    
#     plt.plot(range(past_window_size), past_pm25_test_org_scl[idx], label='PM2.5 (Past Data)', color='blue', alpha=0.7)
    
#     #Plot actual PM2.5 values
#     #plt.plot(actual_labels_original_scale[i], label='Actual PM2.5', color='blue', alpha=0.7)
#     plt.plot(range(past_window_size, past_window_size + future_window_size), 
#              test_actual_labels_original_scale[idx], label='Target (Future Data)', color='orange', alpha=0.7)
    
#     # Plot predicted PM2.5 values
#     #plt.plot(predictions_original_scale[i], label='Predicted PM2.5', color='red', alpha=0.7)
#     plt.plot(range(past_window_size, past_window_size + future_window_size), 
#             test_predictions_original_scale[idx], label='Predicted (Future Data)', color='green', alpha=0.7)
    
    
#     # Set title and labels
#     plt.title(f'Sequence {idx+1}: Predicted vs Actual PM2.5 Values')
#     plt.xlabel('Timestep')
#     plt.ylabel('PM2.5 Value')
    
#     # Show legend and grid
#     plt.legend()
#     plt.grid(True)
    
#     # Display the plot inline
#     display(plt.gcf())  # This will display the plot inline in notebooks

#      # Save the plot to the 'val_folder' with an appropriate file name
#     plot_filename = os.path.join(test_folder, f'sequence_{idx+1}_test_plot.png')
#     plt.savefig(plot_filename)  # Save figure
    
#     # Close the current figure after displaying to avoid overlap
#     plt.close()
#########################################################################################

# Plotting All test predictions at a time with datetime

# Convert year, month, day, hour into datetime format strings for plotting
timestamps = [datetime(int(year), int(month), int(day), int(hour)).strftime('%Y-%m-%d %H')
    for year, month, day, hour in test_pm_dt_original[:, 0, :]]

# Plotting predicted vs actual values
plt.figure(figsize=(14, 7))
plt.plot(test_actual_labels_original_scale, label='Test Actual PM2.5', color='blue', alpha=0.7)
plt.plot(test_predictions_original_scale, label='Test Predicted PM2.5', color='orange', alpha=0.7)
plt.title('Test Predicted vs Test Actual PM2.5 Values')
plt.xlabel('Date-Time')
plt.ylabel('PM2.5 Value')
plt.legend()
plt.grid(True)   

# Set x-axis ticks to display every nth timestamp to avoid overlap
n = 50  # Adjust n to control the frequency of x-axis labels for readability
plt.xticks(range(0, len(timestamps), n), timestamps[::n], rotation=45, ha='right')

# Force Matplotlib to only show the specified ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# Save the plot to the 'val_folder' with an appropriate file name
plot_filename = os.path.join(test_folder, 'Test_pred_plot.png')
plt.savefig(plot_filename, dpi=600) # Save figure

plt.show()

# Close the current figure after displaying to avoid overlap
plt.close()

########################################################################################

# Plotting TEST SET predicitions as sequences for SINGLE-STEP predictor.


# Select 5 random sequences of length 100
sequence_length = 100
num_sequences = 10
total_length = len(test_predictions_original_scale)

# Make sure we can select sequences within the available length
if total_length >= sequence_length:
    random_indices = random.sample(range(total_length - sequence_length), num_sequences)
else:
    raise ValueError("Not enough data points for the requested sequence length.")


# Convert year, month, day, hour into datetime format strings for plotting
timestamps = [datetime(int(year), int(month), int(day), int(hour)).strftime('%Y-%m-%d %H')
              for year, month, day, hour in test_pm_dt_original[:, 0, :]]

# Plot 5 random sequences of length 100
for i, start_idx in enumerate(random_indices):
    
    plt.figure(figsize=(14, 7))
    
    end_idx = start_idx + sequence_length
    
    # Extract date-time labels for the current sequence
    sequence_timestamps = timestamps[start_idx:end_idx]
    
    plt.plot(range(sequence_length), test_actual_labels_original_scale[start_idx:end_idx],
             label=f'Actual PM2.5 (Seq {i+1})', alpha=0.7)
    plt.plot(range(sequence_length), test_predictions_original_scale[start_idx:end_idx],
             label=f'Predicted PM2.5 (Seq {i+1})', alpha=0.7)
    
    # Set title and labels
    plt.title(f'Sequence {i+1}: Predicted vs Actual PM2.5 Values')
    plt.xlabel('Date-time')
    plt.ylabel('PM2.5 Value')
    plt.legend()
    plt.grid(True)
    
    # Set x-axis ticks to display every nth timestamp to avoid overlap
    n = 10  # Adjust n to control the frequency of x-axis labels for readability
    plt.xticks(range(0, sequence_length, n), sequence_timestamps[::n], rotation=45, ha='right')

    # Force Matplotlib to only show the specified ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save each plot with a different filename
    plot_filename = os.path.join(test_folder, f'predicted_vs_actual_sequence_{i+1}.png')
    plt.savefig(plot_filename)
    
    plt.show()
    # Close the plot to avoid overlapping figures
    plt.close()

############################################################################################


# ## Plotting TEST SET predicitions as sequences for MULTI-STEP predictor.

# # Select 5 random sequences of length 100
# sequence_length = 96  # Based on your input data shape
# num_sequences = 5
# future_steps = 4  # Number of future steps to predict
# total_length = len(test_predictions_original_scale)

# # Make sure we can select sequences within the available length
# if total_length >= sequence_length:
#     random_indices = random.sample(range(total_length - sequence_length), num_sequences)
# else:
#     raise ValueError("Not enough data points for the requested sequence length.")


# # Colors for different steps (you can adjust or add more colors if needed)
# prediction_colors = ['orange', 'green', 'red', 'purple']

# # Create and save separate plots for each sequence
# for i, start_idx in enumerate(random_indices):
#     end_idx = start_idx + sequence_length
    
#     # Create a new figure for each sequence
#     plt.figure(figsize=(10, 6))
    
#     # Plot actual and predicted values for the sequence
#     plt.plot(range(sequence_length), test_actual_labels_original_scale[start_idx:end_idx], label=f'Actual PM2.5 (Seq {i+1})', color='blue', alpha=0.7)
#     plt.plot(range(sequence_length), test_predictions_original_scale[start_idx:end_idx], label=f'Predicted PM2.5 (Seq {i+1})', color='orange', alpha=0.7)
    
#     # Set title and labels
#     plt.title(f'Sequence {i+1}: Predicted vs Actual PM2.5 for Multi-Step')
#     plt.xlabel('Time Step')
#     plt.ylabel('PM2.5 Value')
#     plt.legend()
#     plt.grid(True)
    
#     # Save each plot with a higher DPI (e.g., 300) for better quality
#     plot_filename = os.path.join(test_folder, f'predicted_vs_actual_sequence_{i+1}.png')
#     plt.savefig(plot_filename, dpi=300)
    
#     plt.show()
#     # Close the plot to avoid overlapping figures
#     plt.close()



