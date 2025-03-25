
# In[34]:

import os
import torch
import numpy as np
import pandas as pd
import shutil
import random
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset
from Network.Inf_network.model import Informer
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import wandb

# In[35]:

import importlib
import sys
import matplotlib.pyplot as plt

sys.path.append('.')  # Add the parent directory to the Python path
import params_informer
importlib.reload(params_informer)

# In[36]:

print(params_informer.seq_len)

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
config_file_path = './params_informer.py'

# Destination where the config.py file will be copied (inside the study folder)
config_destination_path = os.path.join(study_folder, 'params_informer.py')

# Copy the config.py file to the study folder
shutil.copy(config_file_path, config_destination_path)

print(f'Config file saved to: {config_destination_path}')

#######################################################################################
# WandB variables

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
wandb_name = params_informer.wandb_name
wandb_project = params_informer.wandb_project

wandb_model_name = wandb_name + f'_multivar_{current_time}' + \
            '_bs' + str(params_informer.batch_size) + '_lr' + str(params_informer.lr) + '_ep' + \
            str(params_informer.num_epochs)  +'.pt'
                      
# In[40]:

import sys
sys.path.append('./Data')  # Add the current directory to Python path
import prepare_data_inf
importlib.reload(prepare_data_inf)
from prepare_data_inf import DataPreparer

# In[41]:

# Create an instance of DataPreparer
data_preparer = DataPreparer(data_dir='./01_PM2.5 Chinese Weather data')

# Prepare the data (loads, cleans, splits, and creates tensors) and call the tensors
(train_enc_tensor, train_dec_tensor, train_en_dtstamp_tns, train_dec_dtstamp_tns, train_op_gt,
 val_enc_tensor, val_dec_tensor, val_en_dtstamp_tns, val_dec_dtstamp_tns, val_op_gt,
 test_enc_tensor, test_dec_tensor, test_en_dtstamp_tns, test_dec_dtstamp_tns, test_op_gt,
 test_dec_org_datestamp, test_op_org_datestamp, scaler, column_names, col_ind) = data_preparer.prepare_data()

# Remove the first four column names
columns_without_datetime = column_names[4:]
#-----------------------------------------------------------------------------------

print("Train data tensor shape:", train_enc_tensor.shape)
print("Train labels tensor shape:", train_op_gt.shape)

print("Val data tensor shape:", val_enc_tensor.shape)
print("Val labels tensor shape:", val_op_gt.shape)

print("Test data tensor shape:", test_enc_tensor.shape)
print("Test labels tensor shape:", test_op_gt.shape)

# In[45]:
 
# Prepare the datasets and dataloaders

class InformerDataset(Dataset):
    def __init__(self, enc_x, dec_x, datestamp_enc, datestamp_dec, output_y):
        """
        Custom dataset for Informer.

        Args:
            enc_x (torch.Tensor): Encoder input data
                Shape: (num_samples, seq_len, enc_in)
            dec_x (torch.Tensor): Decoder input data
                Shape: (num_samples, label_len + pred_len, dec_in)
            datestamp_enc (torch.Tensor): Encoder timestamps
                Shape: (num_samples, seq_len, timestamp_dim)
            datestamp_dec (torch.Tensor): Decoder timestamps
                Shape: (num_samples, label_len + pred_len, timestamp_dim)
            output_y (torch.Tensor): Target outputs
                Shape: (num_samples, pred_len, output_dim)
                
        """
        self.enc_x = enc_x.to(torch.float32)
        self.dec_x = dec_x.to(torch.float32)
        self.datestamp_enc = datestamp_enc.to(torch.float32)
        self.datestamp_dec = datestamp_dec.to(torch.float32)
        self.output_y = output_y.to(torch.float32)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.enc_x)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset at the given index.
        
        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple of:
                x_enc (torch.Tensor): Encoder input
                y (torch.Tensor): Target output
                x_mark_enc (torch.Tensor): Encoder timestamp information
                x_mark_dec (torch.Tensor): Decoder timestamp information
        """
        # Encoder inputs
        x_enc = self.enc_x[index]  # Shape: (seq_len, enc_in)
        x_mark_enc = self.datestamp_enc[index]  # Shape: (seq_len, timestamp_dim)

        # Decoder inputs
        x_dec = self.dec_x[index]  # Shape: (label_len + pred_len, dec_in)
        x_mark_dec = self.datestamp_dec[index]  # Shape: (label_len + pred_len, timestamp_dim)

        # Targets
        output_y = self.output_y[index]
        
        return x_enc, x_dec, x_mark_enc, x_mark_dec, output_y

# Create train dataset and dataloader
train_dataset = InformerDataset(
    train_enc_tensor,
    train_dec_tensor,
    train_en_dtstamp_tns,
    train_dec_dtstamp_tns,
    train_op_gt)

# Create Train DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params_informer.batch_size,
                                           shuffle=False, drop_last=True)


#Create Val dataset and dataloader
val_dataset = InformerDataset(
    val_enc_tensor,
    val_dec_tensor,
    val_en_dtstamp_tns,
    val_dec_dtstamp_tns,
    val_op_gt)

# Create Train DataLoader
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params_informer.batch_size,
                                           shuffle=False, drop_last=True)

# Create Test dataset and dataloader
test_dataset = InformerDataset(
    test_enc_tensor,
    test_dec_tensor,
    test_en_dtstamp_tns,
    test_dec_dtstamp_tns,
    test_op_gt)

# Create Test DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params_informer.batch_size,
                                           shuffle=False, drop_last=True)

#----------------------------------------------------------------------------------------------

seq_len = params_informer.seq_len
label_len = params_informer.label_len
pred_len = params_informer.pred_len

#-------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------------------------------
def _process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gt_y):
    """ Forward pass through the model. """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    batch_gt_y = batch_gt_y.float().to(device)

    # Create mask for NaNs (1 for valid values, 0 for NaNs)
    mask = torch.isnan(batch_gt_y)  # Boolean mask (True where NaN, False where valid)
    mask = (~mask).float()  # Convert to float (1 for valid, 0 for NaNs)
    
    # Replace NaNs with 0s in input tensors (so they don't propagate NaNs in computations)
    batch_x = torch.nan_to_num(batch_x, nan=0.0)
    batch_y = torch.nan_to_num(batch_y, nan=0.0)
    batch_gt_y = torch.nan_to_num(batch_gt_y, nan=0.0)

    # Forward pass
    outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    outputs = outputs.squeeze(-1)

    # Apply mask **only for loss calculation**, not for saving predictions
    loss = criterion(outputs * mask, batch_gt_y * mask)  # Masking only in loss
    
    return outputs, batch_gt_y, mask, loss

# In[46]:
    
# This function is useful when we want to calculate loss of only certain features in the dataset 
# (excluding some of the input features for loss caculation) 

column_indices = col_ind
def _process_one_batch_test(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gt_y):
    """ Forward pass through the model during testing, calculating loss only for selected features. """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)
    batch_gt_y = batch_gt_y.float().to(device)

    # Create mask for NaNs (1 for valid values, 0 for NaNs)
    mask = torch.isnan(batch_gt_y)  # Boolean mask (True where NaN, False where valid)
    mask = (~mask).float()  # Convert to float (1 for valid, 0 for NaNs)

    # Replace NaNs with 0s in input tensors
    batch_x = torch.nan_to_num(batch_x, nan=0.0)
    batch_y = torch.nan_to_num(batch_y, nan=0.0)
    batch_gt_y = torch.nan_to_num(batch_gt_y, nan=0.0)

    # Forward pass
    outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    outputs = outputs.squeeze(-1)

    # Select only specific feature indices
    selected_outputs = outputs[:, :, column_indices]  # Selecting specific feature columns
    selected_gt_y = batch_gt_y[:, :, column_indices]  # Selecting same columns in ground truth

    # Apply mask only for the selected features
    selected_mask = mask[:, :, column_indices]

    # Calculate loss only on selected features
    loss = criterion(selected_outputs * selected_mask, selected_gt_y * selected_mask)

    return outputs, batch_gt_y, mask, loss  # Return full outputs but loss for selected features only

# #----------------------------------------------------------------------------------------------
# Define model, loss function, and optimizer

enc_inp = params_informer.enc_inp  # number of features
dec_inp = params_informer.dec_in 
c_out = params_informer.c_out  # output size (1 for regression, could be different for classification)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Informer(enc_inp, dec_inp, c_out, seq_len, label_len, pred_len, device = device).to(device)

# # Loss and optimizer
criterion = nn.L1Loss() # Mean Absolute Error (MAE)

optimizer = optim.Adam(model.parameters(), lr = params_informer.lr)

# In[49]:

#Storing best model based on test loss and train loss    
best_val_loss = float('inf')  # Initialize best test loss as infinity
best_train_loss = float('inf') 

test_checkpoint_path = os.path.join(train_folder, 'best_test_model.pth')  # Path to save the best model    
train_checkpoint_path = os.path.join(train_folder, 'best_train_model.pth')
# Path to save the best train and test losses
train_results_file = os.path.join(train_folder, 'best_losses.txt')            
####################################################################################################

#Train the model

#Intianting WandB
wandb.init(  
      project= wandb_project,    
      config={
        "epochs": params_informer.num_epochs,
        "bs": params_informer.batch_size,
        "lr": params_informer.lr,
        "lookback Horz": params_informer.seq_len,
        "label len": params_informer.label_len,
        })

config = wandb.config
wandb.run.name = wandb_model_name

# Set device (better readability)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = params_informer.num_epochs
patience = params_informer.early_stop_patience  # Number of epochs to wait before stopping if no improvement
early_stopping_counter = 0  # Track epochs without improvement

# Initialize list to store loss values
loss_values = []
val_losses = []   # Store test loss for each 5th epoch

# Add a warm-up phase to skip saving early test evaluations
# Define the number of epochs to skip for saving the model
warmup_epochs = params_informer.warmup_epochs 

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Initialize a list to store predictions
    train_predictions = []
    train_actual_labels = []
    train_masks = []

    for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, batch_gt_y) in enumerate(train_loader):
    
        batch_x = batch_x.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_y = batch_y.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_x_mark = batch_x_mark.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_y_mark = batch_y_mark.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_gt_y = batch_gt_y.to('cuda' if torch.cuda.is_available() else 'cpu')
      
        # Zero the gradients
        optimizer.zero_grad()
        
        pred, true, mask, loss = _process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gt_y)
    
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        train_predictions.append(pred.detach().cpu().numpy())  # Move predictions to CPU and store
        train_actual_labels.append(true.detach().cpu().numpy())
        train_masks.append(mask.detach().cpu().numpy())
              
    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    loss_values.append(avg_loss)  # Store average loss for this epoch
    wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
    
    # Check if we should update training loss checkpoint (every 5 epochs)
    if (epoch + 1) % 5 == 0:
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            print(f'New best training loss: {best_train_loss:.4f}. Saving model based on training loss.')
            torch.save(model.state_dict(), train_checkpoint_path)  # Save the model based on training loss
    
    # VALIDATION LOOP
    
    # ---------- Validating parallely while training (every 5 epochs)------------------------------------------
    if (epoch + 1) % 5 == 0:
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0

        with torch.no_grad():  # Disable gradient calculations for testing
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark, batch_gt_y) in enumerate(val_loader):
            
                batch_x = batch_x.to('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to('cuda' if torch.cuda.is_available() else 'cpu')
                batch_x_mark = batch_x_mark.to('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y_mark = batch_y_mark.to('cuda' if torch.cuda.is_available() else 'cpu')
                batch_gt_y = batch_gt_y.to('cuda' if torch.cuda.is_available() else 'cpu')

                # Forward pass           
                pred, true, mask, loss = _process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gt_y)
            
                running_val_loss += loss.item()               
                
        # Compute average test loss for the epoch
        avg_val_loss = running_val_loss / len(val_loader)
        # avg_test_loss = np.average(running_test_loss)
        val_losses.append(avg_val_loss)
        wandb.log({"test/epoch_loss": avg_val_loss})
        # Print test loss for the current evaluation
        print(f'-- Evaluation after Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}')
        
        # Check if this is the best test loss so far
        # Save the model only after warm-up phase
        if epoch + 1 >= warmup_epochs and avg_val_loss < best_val_loss:    
            best_val_loss = avg_val_loss
            early_stopping_counter = 0  # Reset patience counter
            print(f'New best val loss: {best_val_loss:.4f}. Saving model.')
            torch.save(model.state_dict(), test_checkpoint_path)  # Save model
        else:
           early_stopping_counter += 1
           print(f'No improvement in val loss for {early_stopping_counter}/{patience} epochs.')           
                   
        # Stop training if no improvement for 'patience' epochs
        if early_stopping_counter > patience:
            print("Early stopping triggered. Loading best model and stopping training.")
            model.load_state_dict(torch.load(test_checkpoint_path))  # Restore best model
            break  # Exit training loop   
                
        # Write the train and test losses to the text file in append mode
        with open(train_results_file, 'a') as f:  # Use 'a' for append mode
            f.write(f'Best train Loss (MAE) obtained: {best_train_loss:.4f}\n')
            f.write(f'Best VAL Loss (MAE) obtained: {best_val_loss:.4f}\n')
        
        model.train()  # Switch back to training mode
#-------------------------------------------------------------------------------------------------

# Reverse scaling predictions and actual values
def reverse_scaling(data, scaler, mask):
    # Move tensors to CPU and convert to NumPy if needed
    data = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.array(data)
    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)

    # Ensure shape consistency
    if mask.shape != data.shape:
        mask = np.broadcast_to(mask, data.shape)  # Expand mask if necessary

    # Reshape data for inverse transform: (batch_size * seq_len, num_features)
    original_shape = data.shape  # (batch, seq_len, features)
    data_reshaped = data.reshape(-1, original_shape[-1])

    # Apply inverse scaling
    data_original_scale = scaler.inverse_transform(data_reshaped)

    # Restore NaNs based on mask
    data_original_scale = data_original_scale.reshape(original_shape)
    data_original_scale[mask == 0] = np.nan  # Restore NaNs where they were masked

    return data_original_scale

# Reverse scaling predictions and labels
# Convert list of batches into a single NumPy array
train_predictions = np.concatenate(train_predictions, axis=0)  # Shape: (N, 180, 10)
train_actual_labels = np.concatenate(train_actual_labels, axis=0)  # Shape: (N, 180, 10)
train_masks = np.concatenate(train_masks, axis=0)  # Ensure mask is also concatenated

# Reverse scale both predictions and actual values
train_predictions_original_scale = reverse_scaling(train_predictions, scaler, train_masks)
train_actual_labels_original_scale = reverse_scaling(train_actual_labels, scaler, train_masks)

# In[50]:
# Plot the loss curve after training

lossfig_path = os.path.join(train_folder, 'loss_curve.png')
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(loss_values)+1), loss_values, label='Training Loss')
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

# Randomly select 3 features from the 10 available
num_features = params_informer.enc_inp
num_features_to_plot = 2
num_samples_to_plot = 2
selected_feature_indices = random.sample(range(num_features), num_features_to_plot)

# Get corresponding feature names
selected_feature_names = [columns_without_datetime[idx] for idx in selected_feature_indices]

# Randomly select 10 samples from the 3200 available
num_samples = train_actual_labels_original_scale.shape[0]  # 3200
selected_samples = random.sample(range(1, num_features), num_samples_to_plot)

# Iterate over each sample and each selected feature
for i, sample_idx in enumerate(selected_samples):
    actual_sample = train_actual_labels_original_scale[sample_idx]  # Shape: (180, 10)
    prediction_sample = train_predictions_original_scale[sample_idx]  # Shape: (180, 10)

    for j, (feature_idx, feature_name) in enumerate(zip(selected_feature_indices, selected_feature_names)):
        plt.figure(figsize=(12, 6))

        # Plot actual vs predicted for the given feature
        plt.plot(actual_sample[:, feature_idx], label=f'Actual {feature_name}', color='blue', alpha=0.7)
        plt.plot(prediction_sample[:, feature_idx], label=f'Predicted {feature_name}', color='orange', alpha=0.7)

        plt.title(f'Train Prediction vs Actual (Sample {sample_idx}, Feature: {feature_name})')
        plt.xlabel('Time Step (180)')
        plt.ylabel(feature_name)  # Use feature name for y-axis label
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_filename = os.path.join(train_folder, f'train_pred_sample_{sample_idx}_feature_{feature_name}.png')
        plt.savefig(plot_filename, dpi=600)

        plt.show()
        plt.close()
        
###################################################################################################
###################################################################################################
###################################################################################################

## TESTING LOOP

#Evaluation of the TESTING Set
model.load_state_dict(torch.load(test_checkpoint_path, weights_only = True))
# model.load_state_dict(torch.load(train_checkpoint_path, weights_only = True))

# Set the model to evaluation mode
model.eval()

# Initialize a list to store predictions
test_predictions = []
test_actual_labels = []
test_masks = []
test_past_values = []
test_date_stamp = []

# Initialize total loss for calculating MSE over the validation set
total_test_loss = 0.0

# Path to save the validation results
test_results_file = os.path.join(test_folder, 'test_results.txt')

# Make predictions on the validation dataset using the DataLoader
with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gt_y) in enumerate(test_loader):
    
        batch_x = batch_x.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_y = batch_y.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_x_mark = batch_x_mark.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_y_mark = batch_y_mark.to('cuda' if torch.cuda.is_available() else 'cpu')
        batch_gt_y = batch_gt_y.to('cuda' if torch.cuda.is_available() else 'cpu')
       
        # Forward pass
        pred, true, mask, test_loss = _process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gt_y)              
        
        # # Use the following if we need to calculate loss of only certain features
        # pred, true, mask, test_loss = _process_one_batch_test(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_gt_y)
        
        # Accumulate the total validation loss
        total_test_loss += test_loss.item()
        
        # Store predictions and actual labels for later analysis or metrics calculation
        test_predictions.append(pred.cpu().numpy())  # Move predictions to CPU and store
        test_actual_labels.append(true.cpu().numpy())     # Move labels to CPU and store
        test_masks.append(mask.detach().cpu().numpy())
        test_past_values.append(batch_y.cpu().numpy())
        test_date_stamp.append(batch_y_mark.cpu().numpy())

# Calculate the average MAE over the entire test set
avg_test_loss = total_test_loss / len(test_loader)
print(f'Avg Test Loss before reverse scaling(MAE): {avg_test_loss:.4f}')

# Convert list of batches into a single NumPy array
test_predictions = np.concatenate(test_predictions, axis=0)  # Shape: (N, 180, 10)
test_actual_labels = np.concatenate(test_actual_labels, axis=0)  # Shape: (N, 180, 10)
test_masks = np.concatenate(test_masks, axis=0)  # Ensure mask is also concatenated
test_pastvalues = np.concatenate(test_past_values, axis=0)
test_datestamp = np.concatenate(test_date_stamp, axis=0)

test_predictions_original_scale = reverse_scaling(test_predictions, scaler, np.ones_like(test_masks))  # Do NOT mask predictions
test_actual_labels_original_scale = reverse_scaling(test_actual_labels, scaler, test_masks)  # Mask only ground truth

# Ensure only ground truth remains masked
test_actual_labels_original_scale[test_masks == 0] = np.nan  # Ground truth remains NaN where missing

# Reverse scaling test_pastvalues
# Create a mask where True (1) represents valid values, and False (0) represents NaNs
test_pastvalues_mask = ~np.isnan(test_pastvalues)  # Shape: (N, 365, 10)
test_pastvalues_org_scale = reverse_scaling(test_pastvalues, scaler, test_pastvalues_mask)

# Calculate the Mean Absolute Error (MAE) after reverse scaling
valid_indices = ~np.isnan(test_predictions_original_scale) & ~np.isnan(test_actual_labels_original_scale)
MAE_after_reverse_scaling = np.mean(np.abs(test_predictions_original_scale[valid_indices] - test_actual_labels_original_scale[valid_indices]))

# #-------------------------------------------------------------------------------------------
# # These steps should be included when we want to calcuate loss of only certain features

# # Extract only the selected feature indices for evaluation
# test_predictions_selected = test_predictions_original_scale[:, :, column_indices]
# test_actual_labels_selected = test_actual_labels_original_scale[:, :, column_indices]

# # Compute Mean Absolute Error (MAE) only for the selected features
# valid_indices = ~np.isnan(test_predictions_selected) & ~np.isnan(test_actual_labels_selected)
# MAE_after_reverse_scaling = np.mean(np.abs(test_predictions_selected[valid_indices] - test_actual_labels_selected[valid_indices]))
# #------------------------------------------------------------------------------------------

print(f'Avg Test Loss after reverse scaling (MAE): {MAE_after_reverse_scaling:.4f}')

# Write the validation loss to a text file
with open(test_results_file, 'w') as f:
    f.write(f'Test Loss before reverse scaling(MAE) : {avg_test_loss:.4f}\n')
    f.write(f'Test Loss after reverse scaling(MAE) : {MAE_after_reverse_scaling:.4f}\n')
    f.write(f'Test Predictions Shape: {test_predictions.shape}\n')
    f.write(f'Test Actual Labels Shape: {test_actual_labels.shape}\n')

print(f'Test results saved to: {test_results_file}')        
        
# In[53]:

# Plotting all test predictions in seperate plots for each of the features
test_actual_concat = test_actual_labels_original_scale.reshape(-1, params_informer.enc_inp)  
test_pred_concat = test_predictions_original_scale.reshape(-1, params_informer.enc_inp)

# Plot each feature separately
for feature_idx in range(params_informer.enc_inp):
    plt.figure(figsize=(15, 7))

    # Plot all samples of actual vs predicted for each sample in the features
    plt.plot(test_actual_concat[:, feature_idx], label=f'Actual {columns_without_datetime[feature_idx]}', color='blue', alpha=0.7)
    plt.plot(test_pred_concat[:, feature_idx], label=f'Predicted {columns_without_datetime[feature_idx]}', color='orange', alpha=0.7)

    plt.title(f'Test Predictions vs Actual - {columns_without_datetime[feature_idx]}')
    plt.xlabel('Time Step')
    plt.ylabel(columns_without_datetime[feature_idx])
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = os.path.join(test_folder, f'test_pred_feature_{feature_idx+1}.png')
    plt.savefig(plot_filename, dpi=600)

    plt.show()
    plt.close()
   
################################################################################################
# Visualization of predicted forecast along with past values and ground truth.

# Set seed for reproducibility
random.seed(40)

# Define parameters
num_features = params_informer.enc_inp  # Total number of features
num_samples_to_plot = params_informer.num_samples_to_plot # Number of random samples to plot
total_samples = test_predictions_original_scale.shape[0]  # Total available test samples

# Randomly select samples for plotting
selected_samples = random.sample(range(total_samples), num_samples_to_plot)

# Define the specific features to plot
selected_feature_names = params_informer.feat_for_plotting

# Get corresponding indices from columns_without_datetime
selected_feature_indices = [columns_without_datetime.index(feature) for feature in selected_feature_names]

# Iterate over selected samples and features
for sample_idx in selected_samples:
    # Extract relevant data for the selected sample
    past_values = test_pastvalues_org_scale[sample_idx]
    actual_values = test_actual_labels_original_scale[sample_idx]  
    predicted_values = test_predictions_original_scale[sample_idx] 

    # Insert the first actual value at index 91 in past_values
    past_values_extended = np.vstack((past_values[:params_informer.label_len, :], actual_values[0, :], past_values[params_informer.label_len:, :])) 
    
    # Extract past_values_modified (0 to 91 indices)
    past_values_modified = past_values_extended[:params_informer.label_len+1, :] 
    
    for feature_idx, feature_name in zip(selected_feature_indices, selected_feature_names):
        plt.figure(figsize=(12, 6))

        # Define time axes 
        time_steps_past = np.arange(1, params_informer.label_len+1)  # Past time steps
        time_steps_future = np.arange(params_informer.label_len+1, params_informer.label_len+1 + params_informer.pred_len)
        
        #----------------------------------------------------------------------------------
        # # Code for extracting datetime stamps for plotting
        # # Extract the correct timestamps for the selected sample
        # time_steps_past = test_dec_org_datestamp[sample_idx, 1:params_informer.label_len+1, 0]      
        # time_steps_future = test_op_org_datestamp[sample_idx, :, 0]  # all gt timestamps
        #----------------------------------------------------------------------------------
        
        # Create mask for valid (non-NaN) past values
        valid_past_mask = ~np.isnan(past_values_modified[1:, feature_idx])            
        
        # Conditionally switch between scatter and plot
        if params_informer.nan_fraction > 0.75:
            # Use scatter for highly sparse data
            plt.scatter(time_steps_past[valid_past_mask], 
                        past_values_modified[1:, feature_idx][valid_past_mask], 
                        label=f'Past {feature_name}', color='blue', alpha=0.7, marker='o')
        else:
            # Use line plot for less sparse data
            plt.plot(time_steps_past, past_values_modified[1:, feature_idx], 
                     label=f'Past {feature_name}', color='blue', alpha=0.7)

        # Plot actual future values
        plt.plot(time_steps_future, actual_values[:, feature_idx], 
                 label=f'Actual {feature_name}', color='green', alpha=0.7)

        # Plot predicted future values (dashed line)
        plt.plot(time_steps_future, predicted_values[:, feature_idx], 
                 label=f'Predicted {feature_name}', color='red', linestyle='dashed', alpha=0.9)

        #--------------------------------------------------------------------------------------------
        # # Code to Include Date Time Stamps on the x-axis
        # # Select every 30th timestamp for labeling
        # tick_indices = np.arange(0, len(time_steps_past) + len(time_steps_future), 31)
        # tick_labels = np.concatenate([time_steps_past, time_steps_future])[tick_indices]
               
        # # Convert timestamps to datetime format
        # formatted_dates = [pd.to_datetime(ts) for ts in tick_labels]
        
        # # Format into two-line strings: "YYYY-MM-DD\nHH:MM"
        # formatted_labels = [date.strftime('%Y-%m-%d\n%H:%M') for date in formatted_dates]
        
        # # Apply formatted labels
        # plt.xticks(tick_labels, formatted_labels, rotation=45, ha="center")  # Center-align
        #--------------------------------------------------------------------------------------------
        
        # Labels and title
        plt.title(f'Test Prediction vs Actual (Sample {sample_idx}, Feature: {feature_name})')
        plt.xlabel('Time Steps')
        plt.ylabel(feature_name)
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_filename = os.path.join(test_folder, f'test_pred_sample_{sample_idx}_feature_{feature_name}.png')
        plt.savefig(plot_filename, dpi=600)

        plt.show()
        plt.close()
