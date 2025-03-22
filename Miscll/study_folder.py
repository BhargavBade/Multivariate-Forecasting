import os
from datetime import datetime

# Create a folder named with the current DateTime in the Results folder
def create_study_folder(base_dir='./Results'):
    # Get current DateTime and format it for the folder name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_folder = os.path.join(base_dir, f'Informer_Study_{current_time}')
    
    # Create the folder if it doesn't exist
    if not os.path.exists(study_folder):
        os.makedirs(study_folder)

    # Create subfolders for 'Train', 'Test', and 'Val' within the study folder
    train_folder = os.path.join(study_folder, 'Train')
    test_folder = os.path.join(study_folder, 'Test')
    val_folder = os.path.join(study_folder, 'Val')
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    return study_folder, train_folder, test_folder, val_folder
    
