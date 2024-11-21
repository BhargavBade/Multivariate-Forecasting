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
    