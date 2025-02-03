# Set seed for reproducibility
random.seed(42)

# Define parameters
past_window = params.n_steps_in+1  # Length of past PM2.5 data
future_steps = params.n_steps_out  # Length of predicted & actual data
total_window = past_window + future_steps  # Total length in plot
num_plots = 5  # Number of plots to generate
total_length = len(test_predictions_original_scale)  # Ensure valid sampling range

# Ensure we have enough data points
if total_length < total_window:
    raise ValueError("Not enough test data points for the requested plot length.")

# Randomly select starting points within valid range
random_indices = random.sample(range(total_length - past_window), num_plots)

# Create plots
for i, past_start_idx in enumerate(random_indices):
    plt.figure(figsize=(10, 5))
    
    # Define past & future indices
    past_end_idx = past_start_idx + past_window  # End of past window
    future_start_idx = past_end_idx-1 # Align future data with past sequence
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
    plt.plot(time_steps_future, actual_future_pm25, label="Actual PM2.5", color="green", linestyle="dashed", alpha=0.7)
    
    # # Create a combined timeline (past + future) for smooth plotting
    # combined_past_and_future = np.concatenate((past_pm25, actual_future_pm25))
    # combined_time_steps = np.concatenate((time_steps_past, time_steps_future))

    # # Plot combined past and actual future PM2.5 as a continuous line (past in blue, future in green)
    # plt.plot(combined_time_steps, combined_past_and_future, label="Past + Actual Future PM2.5", color="blue", alpha=0.7)
    
    # Plot predicted future PM2.5 values
    plt.plot(time_steps_future, predicted_future_pm25, label="Predicted PM2.5", color="red", linestyle="dotted", alpha=0.7)

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