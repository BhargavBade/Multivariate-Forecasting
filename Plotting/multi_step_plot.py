

# ## Plotting TEST SET predicitions as sequences for MULTI-STEP predictor.

# # Select 5 random sequences of length 100
# sequence_length = 400  # Based on your input data shape
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
