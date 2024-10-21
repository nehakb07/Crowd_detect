from ultralytics import YOLO
import random
import torch  # Import torch to check for GPU availability

# Initialize the YOLO model
model = YOLO('C:/Users/Neha KB/Desktop/custom/c_train/yolov8n.pt')  # Adjust to the appropriate YOLOv8 model path

# Define the search space for random search
search_space = {
    'lr0': [0.0001, 0.001, 0.01],  # Learning rate
    'optimizer': ['SGD', 'AdamW'],  # Optimizer choices
    'momentum': [0.8, 0.9, 0.95, 0.99],  # Momentum values
    'weight_decay': [0.0001, 0.001, 0.01],  # Weight decay values
    'cos_lr': [True, False]  # Cosine learning rate scheduler flag
}

# Number of random samples (iterations)
num_samples = 20  # Adjust based on time/resources

# Variables to track the best hyperparameters and performance metrics
best_map = 0  # Best mAP
best_f1 = 0  # Best F1 score
best_hyperparams = None  # To store the best hyperparameters

# Check if CUDA is available and set device accordingly
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Loop through random combinations of hyperparameters
for i in range(num_samples):
    # Randomly select hyperparameters from the search space
    hyperparams = {k: random.choice(v) for k, v in search_space.items()}
    
    # Define the training parameters with the sampled hyperparameters
    training_params = {
        'data': 'C:/Users/Neha KB/Desktop/custom/c_train/data.yaml',  # Path to your dataset YAML file
        'epochs': 50,  # Number of epochs, set lower for tuning
        'lr0': hyperparams['lr0'],  # Learning rate
        'batch': 4,  # Fixed batch size due to GPU constraints
        'optimizer': hyperparams['optimizer'],  # Optimizer
        'momentum': hyperparams['momentum'],  # Momentum
        'weight_decay': hyperparams['weight_decay'],  # Weight decay
        'imgsz': 640,  # Fixed image size at 640x640
        'cos_lr': hyperparams['cos_lr'],  # Cosine LR scheduler flag
        'device': device,  # Use detected device
        'project': 'crowd_detection_project',  # Save results in a specific folder
        'name': f'random_search_trial_{i}',  # Unique name for each trial
        'exist_ok': True  # Overwrite existing results if they exist
    }

    # Train the model with the current set of hyperparameters
    try:
        results = model.train(**training_params)
        
        # Get performance metrics from the results
        current_map = results.metrics['mAP50-95']  # Assuming results.metrics contains mAP50-95
        current_f1 = results.metrics['F1']  # Assuming results.metrics contains F1 score
        
        print(f"Trial {i + 1}/{num_samples}: mAP={current_map}, F1={current_f1}")
        print(f"Hyperparameters: {hyperparams}")

        # Update best hyperparameters if current mAP is better
        if current_map > best_map:
            best_map = current_map
            best_f1 = current_f1
            best_hyperparams = hyperparams

    except Exception as e:
        print(f"Error occurred during trial {i + 1}: {str(e)}")

# Print the best hyperparameters and performance after all trials
print(f"Best mAP: {best_map}, Best F1: {best_f1}")
print(f"Best Hyperparameters: {best_hyperparams}")
