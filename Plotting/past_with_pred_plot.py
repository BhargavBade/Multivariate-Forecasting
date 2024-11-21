# In[ ]:

# Val set plotting with past PM2.5 and predicted PM2.5    

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


# In[ ]:
# Test set plotting with past PM2.5 and predicted PM2.5

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


