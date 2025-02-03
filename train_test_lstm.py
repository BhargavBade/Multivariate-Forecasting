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
import wandb

# In[35]:

import importlib
import sys
import matplotlib.pyplot as plt

sys.path.append('.')  # Add the parent directory to the Python path
import params_lstm
importlib.reload(params_lstm)

# In[36]:

#print(params.window_size)
print(params_lstm.n_steps_in)


# In[37]:

print(params_lstm.hidden_size)


# In[38]:


#Study Folder creation for saving Results and Plots
sys.path.append('./Miscll') 
import study_folder
importlib.reload(study_folder)
study_folder, train_folder, test_folder, val_folder = study_folder.create_study_folder()
print(study_folder)
print(train_folder)

# In[39]:

#saving the params file used for this study in this study folder

# Path to the config.py file in your project directory
config_file_path = './params_lstm.py'

# Destination where the config.py file will be copied (inside the study folder)
config_destination_path = os.path.join(study_folder, 'params_lstm.py')

# Copy the config.py file to the study folder
shutil.copy(config_file_path, config_destination_path)

print(f'Config file saved to: {config_destination_path}')


#######################################################################################
# WandB VARIABLES

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb_name = params_lstm.wandb_name
wandb_project = params_lstm.wandb_project

wandb_model_name = wandb_name + f'_multivar_{current_time}' + \
            '_bs' + str(params_lstm.batch_size) + '_lr' + str(params_lstm.lr) + '_ep' + \
            str(params_lstm.num_epochs)  +'.pt'
                      
# In[40]:

import sys
sys.path.append('./Data')  # Add the current directory to Python path
# import prepare_data01
# importlib.reload(prepare_data01)
# from prepare_data01 import DataPreparer
############################################################################### 

import prepare_data_lstm
importlib.reload(prepare_data_lstm)
from prepare_data_lstm import DataPreparer

# In[41]:


# Create an instance of DataPreparer
data_preparer = DataPreparer(data_dir='./01_PM2.5 Chinese Weather data')


# In[42]:

# Prepare the data (loads, cleans, splits, and creates tensors)

data_preparer.prepare_data()

# Get the tensors
(train_data_tensor, train_labels_tensor, past_pm25_train, val_data_tensor, 
 val_labels_tensor, past_pm25_val, val_pm_dt, test_data_tensor, test_labels_tensor, 
 past_pm25_test, test_pm_dt, scaler, pm_index) = data_preparer.get_tensors()


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
    def __init__(self, data, labels, past_pm):
        self.data = data.to(torch.float32)
        self.labels = labels.to(torch.float32)
        self.past_pm = past_pm.to(torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.past_pm[idx]

# Train Data and Train Labels
X_train = train_data_tensor.to(torch.float32)
y_train = train_labels_tensor.to(torch.float32)
past_pm_train = past_pm25_train.to(torch.float32)

# Train Dataset and Train Dataloader
train_dataset = TimeSeriesDataset(X_train, y_train, past_pm_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params_lstm.batch_size, 
                                           shuffle = False, drop_last=True)

# Val Data and Val Labels
X_val = val_data_tensor.to(torch.float32)
y_val = val_labels_tensor.to(torch.float32)
past_pm_val = past_pm25_val.to(torch.float32)

# Val Dataset and Val Dataloader
val_dataset = TimeSeriesDataset(X_val, y_val, past_pm_val)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size = params_lstm.batch_size, 
                                         shuffle=False, drop_last=True)

# Test Data and Test Labels
X_test = test_data_tensor.to(torch.float32)
y_test = test_labels_tensor.to(torch.float32)
past_pm_test = past_pm25_test.to(torch.float32)

# Test Dataset and Test Dataloader
test_dataset = TimeSeriesDataset(X_test, y_test, past_pm_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params_lstm.batch_size, 
                                          shuffle=False, drop_last=True)


# In[46]:

# Define model, loss function, and optimizer

input_size = params_lstm.input_size  # number of features
hidden_size = params_lstm.hidden_size  # number of hidden units in LSTM
output_size = params_lstm.output_size  # output size (1 for regression, could be different for classification)
num_layers = params_lstm.num_layers  # number of LSTM layers

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.MSELoss()
# criterion = nn.L1Loss() # Mean Absolute Error (MAE)
optimizer = optim.Adam(model.parameters(), lr = params_lstm.lr)

# In[49]:

#Storing best model based on test loss    
best_test_loss = float('inf')  # Initialize best test loss as infinity
checkpoint_path = os.path.join(train_folder, 'best_model.pth')  # Path to save the best model    

# Path to save the training results
train_results_file = os.path.join(train_folder, 'best_test_loss.txt')
            
####################################################################################################

#Train the model

wandb.init(  
      project= wandb_project,    
      config={
        "epochs": params_lstm.num_epochs,
        "bs": params_lstm.batch_size,
        "lr": params_lstm.lr,
        "lookback Horz": params_lstm.n_steps_in,
        "num_layers": params_lstm.num_layers,
        "latent size": params_lstm.hidden_size,
        })

config = wandb.config
wandb.run.name = wandb_model_name

# Set device (better readability)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = params_lstm.num_epochs

# Initialize list to store loss values
loss_values = []
test_losses = []   # Store test loss for each 5th epoch

# Initialize a list to store predictions
train_predictions = []
train_actual_labels = []

warmup_epochs = 15

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data, labels, batch_past_pm in train_loader:
        data, labels, batch_past_pm = (data.to('cuda' if torch.cuda.is_available() else 'cpu'), 
                                      labels.to('cuda' if torch.cuda.is_available() else 'cpu'),
                                      batch_past_pm.to('cuda' if torch.cuda.is_available() else 'cpu'))

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)

        # Compute the loss (squeeze the outputs so they match the shape of labels)
        # loss = criterion(outputs, labels) #MSE Loss
        
       
        loss = torch.sqrt(criterion(outputs, labels)) #RMSE Loss using torch.nn
        
        # # Calculate RMSE manually
        # squared_diff = (outputs - labels) ** 2  # Element-wise square of differences
        # mse = torch.mean(squared_diff)          # Mean of squared differences
        # loss = torch.sqrt(mse)                  # Square root of MSE
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        train_predictions.append(outputs.detach().cpu().numpy())  # Move predictions to CPU and store
        train_actual_labels.append(labels.detach().cpu().numpy())
        
        
    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    loss_values.append(avg_loss)  # Store average loss for this epoch
    wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # -------- Testing Phase (every 5 epochs) --------
    if (epoch + 1) % 5 == 0:
        model.eval()  # Set the model to evaluation mode
        running_test_loss = 0.0

        with torch.no_grad():  # Disable gradient calculations for testing
            for data, labels, batch_past_pm in test_loader:
                data, labels, batch_past_pm = (data.to('cuda' if torch.cuda.is_available() else 'cpu'), 
                                              labels.to('cuda' if torch.cuda.is_available() else 'cpu'),
                                              batch_past_pm.to('cuda' if torch.cuda.is_available() else 'cpu'))
                
                # Forward pass
                outputs = model(data)

                # Compute the loss
                # loss = criterion(outputs, labels) #MSE Loss
                
                
                loss = torch.sqrt(criterion(outputs, labels)) #RMSE Loss
                
                # # Calculate RMSE manually
                # squared_diff = (outputs - labels) ** 2  # Element-wise square of differences
                # mse = torch.mean(squared_diff)          # Mean of squared differences
                # loss = torch.sqrt(mse)                  # Square root of MSE

                running_test_loss += loss.item()

        # Compute average test loss for the epoch
        avg_test_loss = running_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        wandb.log({"test/epoch_loss": avg_test_loss})
        # Print test loss for the current evaluation
        print(f'-- Evaluation after Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}')
        
        # Check if this is the best test loss so far
        # if avg_test_loss < best_test_loss:
        if epoch + 1 > warmup_epochs and avg_test_loss < best_test_loss:     
            best_test_loss = avg_test_loss
            print(f'New best test loss: {best_test_loss:.4f}. Saving model.')
            torch.save(model.state_dict(), checkpoint_path)  # Save model
        
        # Write the validation loss to a text file
        with open(train_results_file, 'w') as f:
            f.write(f'Best test Loss (MSE) obtained: {best_test_loss:.4f}\n')
        
        
        model.train()  # Switch back to training mode


# Concatenate, squeeze, and flatten predictions and labels into 1D arrays
train_predictions = np.concatenate(train_predictions, axis=0).squeeze().flatten()
train_actual_labels = np.concatenate(train_actual_labels, axis=0).squeeze().flatten()

# Reverse scaling the prediction PM and actual PM values
# Access min and max values for reverse scaling
mins = scaler.data_min_
maxs = scaler.data_max_

# Reverse scaling for PM predictions and labels
train_predictions_original_scale = train_predictions * (maxs[pm_index] - mins[pm_index]) + mins[pm_index]
train_actual_labels_original_scale = train_actual_labels * (maxs[pm_index] - mins[pm_index]) + mins[pm_index]
 
# In[50]:
# Plot the loss curve after training

lossfig_path = os.path.join(train_folder, 'loss_curve.png')
plt.figure(figsize=(10, 8))
plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(lossfig_path) 
plt.show()
plt.close()

#------------------------------------------------------------------------------------------

# Plot the trained data and predictions after finishing the training

chunk_size = 500  # Number of samples per chunk
total_chunks = len(train_actual_labels) // chunk_size + (1 if len(train_actual_labels) % chunk_size != 0 else 0)
num_chunks_to_plot = 10  # Number of chunks to randomly select

# Randomly select 10 unique chunk indices
random_chunk_indices = random.sample(range(total_chunks), min(num_chunks_to_plot, total_chunks))

# Ensure the output folder exists
os.makedirs(train_folder, exist_ok=True)

# Plot each randomly selected chunk
for i, chunk_idx in enumerate(random_chunk_indices):
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size

    # Slice the data for the current chunk
    actual_chunk = train_actual_labels_original_scale[start_idx:end_idx]
    prediction_chunk = train_predictions_original_scale[start_idx:end_idx]

    # Plot the current chunk
    plt.figure(figsize=(14, 7))
    plt.plot(actual_chunk, label='Actual PM2.5', color='blue', alpha=0.7)
    plt.plot(prediction_chunk, label='Predicted PM2.5', color='orange', alpha=0.7)
    plt.title(f'Train Predicted vs Train Actual PM2.5 Values (Chunk {i + 1})')
    plt.xlabel('Sample Index')
    plt.ylabel('PM2.5 Value')
    plt.legend()
    plt.grid(True)

    # Save the plot for the current chunk
    plot_filename = os.path.join(train_folder, f'train_pred_chunk_{i + 1}.png')
    plt.savefig(plot_filename, dpi=600)  # Save figure

    plt.show()

    # Close the current figure after displaying to avoid overlap
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
    for data, labels, batch_past_pm in val_loader:
        data, labels, batch_past_pm = (data.to('cuda' if torch.cuda.is_available() else 'cpu'), 
                                       labels.to('cuda' if torch.cuda.is_available() else 'cpu'),
                                       batch_past_pm.to('cuda' if torch.cuda.is_available() else 'cpu'))
        # Forward pass to get predictions
        predictions = model(data)
        
        # Compute the validation loss (MSE)
        # val_loss = criterion(predictions, labels) #MSE Loss
        
        val_loss = torch.sqrt(criterion(predictions, labels)) #RMSE Loss
        
        # Accumulate the total validation loss
        total_val_loss += val_loss.item()
        
        # Store predictions and actual labels for later analysis or metrics calculation
        val_predictions.append(predictions.cpu().numpy())  # Move predictions to CPU and store
        val_actual_labels.append(labels.cpu().numpy())     # Move labels to CPU and store

# Calculate the average MSE over the entire validation set
avg_val_loss = total_val_loss / len(val_loader)
print(f'Validation Loss (RMSE): {avg_val_loss:.4f}')

# Concatenate all predictions and actual labels into a single array
val_predictions = np.concatenate(val_predictions, axis=0)
val_actual_labels = np.concatenate(val_actual_labels, axis=0)

# Write the validation loss to a text file
with open(val_results_file, 'w') as f:
    f.write(f'Validation Loss (RMSE): {avg_val_loss:.4f}\n')
    f.write(f'Validation Predictions Shape: {val_predictions.shape}\n')
    f.write(f'Validation Actual Labels Shape: {val_actual_labels.shape}\n')

print(f'Validation results saved to: {val_results_file}')        
        
# In[53]:

# Plotting the Validation Predictions vs Actual PM2.5 Values
plt.figure(figsize=(14, 7))
plt.plot(val_actual_labels, label='Val Actual PM2.5', color='blue', alpha=0.7)
plt.plot(val_predictions, label='Val Predicted PM2.5', color='orange', alpha=0.7)
plt.title('Val Predicted vs Val Actual PM2.5 Values')
plt.xlabel('Date-Time')
plt.ylabel('PM2.5 Value')
plt.legend()
plt.grid(True)

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
test_past_pm = []

# Initialize total loss for calculating MSE over the validation set
total_test_loss = 0.0

# Path to save the validation results
test_results_file = os.path.join(test_folder, 'test_results.txt')

# Make predictions on the validation dataset using the DataLoader
with torch.no_grad():
    for data, labels, batch_past_pm in test_loader:
        #data_tensor, _ = data  # We only need the features for prediction
        #predictions = model(data_tensor.to('cuda' if torch.cuda.is_available() else 'cpu'))
        data, labels, batch_past_pm =(data.to('cuda' if torch.cuda.is_available() else 'cpu'), 
                                      labels.to('cuda' if torch.cuda.is_available() else 'cpu'),
                                      batch_past_pm.to('cuda' if torch.cuda.is_available() else 'cpu'))

        # Forward pass to get predictions
        predictions = model(data)
        
        # Compute the validation loss (MSE)

        # test_loss = criterion(predictions, labels) #MSE Loss
        
        test_loss = torch.sqrt(criterion(predictions, labels)) #RMSE Loss
        
        # # Calculate RMSE manually
        # squared_diff = (predictions - labels) ** 2  # Element-wise square of differences
        # mse = torch.mean(squared_diff)          # Mean of squared differences
        # test_loss = torch.sqrt(mse)                  # Square root of MSE
        
        # Accumulate the total validation loss
        total_test_loss += test_loss.item()
        
        # Store predictions and actual labels for later analysis or metrics calculation
        test_predictions.append(predictions.cpu().numpy())  # Move predictions to CPU and store
        test_actual_labels.append(labels.cpu().numpy())     # Move labels to CPU and store
        test_past_pm.append(batch_past_pm.cpu().numpy())

# Calculate the average MAE over the entire test set
avg_test_loss = total_test_loss / len(test_loader)
print(f'Avg Test Loss before reverse scaling(RMSE): {avg_test_loss:.4f}')

# Concatenate all predictions and actual labels into a single array
test_predictions = np.concatenate(test_predictions, axis=0).flatten()
test_actual_labels = np.concatenate(test_actual_labels, axis=0).flatten()
test_past_pm = np.concatenate(test_past_pm, axis=0).flatten()

# Covert past_pm_25 into an array
past_pm25_test2 = past_pm25_test.numpy().flatten()

# Reverse scaling the prediction PM and actual PM values
# Access min and max values for reverse scaling
mins = scaler.data_min_
maxs = scaler.data_max_

# Reverse scaling for PM predictions and labels
test_predictions_original_scale = test_predictions * (maxs[pm_index] - mins[pm_index]) + mins[pm_index]
test_actual_labels_original_scale = test_actual_labels * (maxs[pm_index] - mins[pm_index]) + mins[pm_index]
test_past_pm_original = test_past_pm * (maxs[pm_index] - mins[pm_index]) + mins[pm_index]

test_actual_org_scl2 = past_pm25_test2 * (maxs[pm_index] - mins[pm_index]) + mins[pm_index]

# Calculate the Mean Absolute Error (MAE) after reverse scaling
MAE_after_reverse_scaling = np.mean(np.abs(test_predictions_original_scale - test_actual_labels_original_scale))
print(f'Avg Test Loss after reverse scaling (RMSE): {MAE_after_reverse_scaling:.4f}')

# Write the validation loss to a text file
with open(test_results_file, 'w') as f:
    f.write(f'Test Loss before reverse scaling(RMSE) : {avg_test_loss:.4f}\n')
    f.write(f'Test Loss after reverse scaling(RMSE) : {MAE_after_reverse_scaling:.4f}\n')
    f.write(f'Test Predictions Shape: {test_predictions.shape}\n')
    f.write(f'Test Actual Labels Shape: {test_actual_labels.shape}\n')

print(f'Test results saved to: {test_results_file}')        
        
# In[53]:

# Plotting All test predictions at a time
# Plotting predicted vs actual values
plt.figure(figsize=(14, 7))
plt.plot(test_actual_labels_original_scale, label='Test Actual PM2.5', color='blue', alpha=0.7)
plt.plot(test_predictions_original_scale, label='Test Predicted PM2.5', color='orange', alpha=0.7)
plt.title('Test Predicted vs Test Actual PM2.5 Values')
plt.xlabel('Date-Time')
plt.ylabel('PM2.5 Value')
plt.legend()
plt.grid(True)   

# Save the plot to the 'val_folder' with an appropriate file name
plot_filename = os.path.join(test_folder, 'Test_pred_plot.png')
plt.savefig(plot_filename, dpi=600) # Save figure

plt.show()

# Close the current figure after displaying to avoid overlap
plt.close()

########################################################################################
########################################################################################

# Plotting TEST SET predicitions as sequences for SINGLE-STEP predictor.

# Set a fixed random seed for reproducibility
random_seed = 42
random.seed(random_seed)

# Select 5 random sequences of length 100
sequence_length = 500
num_sequences = 10
total_length1 = len(test_predictions)

# Make sure we can select sequences within the available length
if total_length1 >= sequence_length:
    random_indices = random.sample(range(total_length1 - sequence_length), num_sequences)
else:
    raise ValueError("Not enough data points for the requested sequence length.")

# Plot 5 random sequences of length 100
for i, start_idx in enumerate(random_indices):
    
    plt.figure(figsize=(8, 6))
    
    end_idx = start_idx + sequence_length
      
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
    
    # Save each plot with a different filename
    plot_filename = os.path.join(test_folder, f'predicted_vs_actual_sequence_{i+1}.png')
    plt.savefig(plot_filename)
    
    plt.show()
    # Close the plot to avoid overlapping figures
    plt.close()

###############################################################################################

# Visualization of predicted forecast along with past values and ground truth.

# Set seed for reproducibility
random.seed(42)

# Define parameters
past_window = params_lstm.n_steps_in+1  # Length of past PM2.5 data
future_steps = params_lstm.n_steps_out  # Length of predicted & actual data
total_window = past_window + future_steps  # Total length in plot
num_plots = 15  # Number of plots to generate
total_length = len(test_predictions_original_scale)  # Ensure valid sampling range

# Ensure we have enough data points
if total_length < (num_plots * future_steps):
    raise ValueError("Not enough test data points for the requested plot length.")

# Sequential start points
past_start_indices = np.arange(0, num_plots * past_window, past_window)
forecast_start_indices = np.arange(0, num_plots * future_steps, future_steps)

# # Create plots
# for i, start_idx in enumerate(past_start_indices):
#     plt.figure(figsize=(10, 5))

# Create plots
for i, (past_start_idx, future_start_idx) in enumerate(zip(past_start_indices, forecast_start_indices)):
    plt.figure(figsize=(8, 6))
    
    # Define past & future indices
    # past_start_idx = start_idx
    past_end_idx = past_start_idx + past_window  # End of past window
    # future_start_idx = start_idx # Align future data with past sequence
    future_end_idx = future_start_idx + future_steps # End of future window
    
    # Extract past PM2.5 values (196 steps)
    past_pm25 = test_past_pm_original[past_start_idx:past_end_idx]

    # Extract future actual PM2.5 values (24 steps)
    actual_future_pm25 = test_actual_labels_original_scale[future_start_idx:future_end_idx]

    # Extract future predicted PM2.5 values (24 steps)
    predicted_future_pm25 = test_predictions_original_scale[future_start_idx:future_end_idx]

    # Ensure correct shapes
    assert len(past_pm25) == past_window, f"Past PM2.5 length mismatch: {len(past_pm25)}"
    assert len(actual_future_pm25) == future_steps, f"Actual future PM2.5 length mismatch: {len(actual_future_pm25)}"
    assert len(predicted_future_pm25) == future_steps, f"Predicted future PM2.5 length mismatch: {len(predicted_future_pm25)}"

    # Create timeline for x-axis
    time_steps_past = np.arange(0, past_window)
    time_steps_future = np.arange(past_window, total_window)

    # Plot past PM2.5 values
    plt.plot(time_steps_past, past_pm25, label="Past PM2.5", color="blue", alpha=0.7)

    # Plot actual future PM2.5 values
    plt.plot(time_steps_future, actual_future_pm25, label="Actual PM2.5", color="green", alpha=0.7)
    
    # Plot predicted future PM2.5 values
    plt.plot(time_steps_future, predicted_future_pm25, label="Predicted PM2.5", color="red", linestyle="dashed", alpha=0.9)

    # Labels and title
    plt.title(f"Forecasting Visualization - Sample {i+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("PM2.5 Value")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_filename = os.path.join(test_folder, f"forecast_plot_{i+1}.png")
    plt.savefig(plot_filename)

    # Show and close figure
    plt.show()
    plt.close()