from ultralytics import YOLO
import random
import torch
import pandas as pd

# Initialize the YOLO model
model = YOLO('C:/Users/Neha KB/Desktop/custom/yolov8n.pt')  # Path to YOLOv8n model

# Load the previous best model weights
model.load('C:/Users/Neha KB/Desktop/custom/random_search_trial_2/random_search_trial_0/weights/best.pt')  # Path to your best model

# Define the search space for random search
search_space = {
    'lr0': [0.0001, 0.001, 0.01],  # Learning rates
    'optimizer': ['SGD', 'AdamW'],  # Optimizers
    'momentum': [0.9, 0.95, 0.99],  # Momentum values
    'weight_decay': [0.0001, 0.001, 0.01],  # Weight decay values
}

# Number of random samples to try
num_samples = 50  # Adjust this number based on your needs

# Metrics thresholds for performance
desired_recall = 0.7  # Desired recall threshold
desired_f1 = 0.7  # Desired F1 score threshold

# Check if CUDA is available and set device accordingly
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=['Trial', 'Epoch', 'Recall', 'F1', 'Hyperparameters'])

# Loop through random combinations of hyperparameters until the metrics are achieved
for i in range(num_samples):
    # Randomly select hyperparameters from the search space
    hyperparams = {k: random.choice(v) for k, v in search_space.items()}
    
    # Define the training parameters with the sampled hyperparameters
    training_params = {
        'data': 'C:/Users/Neha KB/Desktop/custom/c_train/data.yaml',  # Path to your dataset YAML file
        'epochs': 150,  # Fixed number of epochs to continue training
        'lr0': hyperparams['lr0'],  # Learning rate
        'batch': 4,  # Batch size
        'optimizer': hyperparams['optimizer'],  # Optimizer
        'momentum': hyperparams['momentum'],  # Momentum
        'weight_decay': hyperparams['weight_decay'],  # Weight decay
        'imgsz': 640,  # Image size
        'device': device,  # Use detected device
        'project': 'crowd_detection_project',  # Save results in a specific folder
        'name': f'random_search_trial_{i}',  # Unique name for each trial
        'exist_ok': True,  # Overwrite existing results if they exist
    }

    # Train the model with the current set of hyperparameters
    try:
        results = model.train(**training_params)
        
        # Get performance metrics for each epoch
        for epoch in range(len(results.metrics['recall'])):  # Assuming metrics has a recall list
            current_recall = results.metrics['recall'][epoch]
            current_f1 = results.metrics['F1'][epoch]
            
            # Store the results in the DataFrame
            results_df = results_df.append({
                'Trial': i + 1,
                'Epoch': epoch + 1,
                'Recall': current_recall,
                'F1': current_f1,
                'Hyperparameters': hyperparams
            }, ignore_index=True)

            # Log the performance metrics
            print(f"Trial {i + 1}, Epoch {epoch + 1}: Recall={current_recall}, F1={current_f1}, Hyperparameters={hyperparams}")

    except Exception as e:
        print(f"Error occurred during trial {i + 1}: {str(e)}")

# Save the results to a CSV file
results_df.to_csv('hyperparameter_tuning_results.csv', index=False)

# Print a message if the loop completes without achieving the target
print("Training completed. All metrics have been saved to hyperparameter_tuning_results.csv.")
