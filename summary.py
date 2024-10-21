import pandas as pd

# Path to your CSV file
csv_file_path = 'C:/Users/Neha KB/Desktop/custom/random_search_trial_2/random_search_trial_0/results.csv'

# Read the CSV file
try:
    df = pd.read_csv(csv_file_path)
    
    # Check if the DataFrame is empty
    if df.empty:
        print("The CSV file is empty.")
    else:
        # Display general information about the DataFrame
        print("DataFrame Info:")
        print(df.info())
        
        # Display summary statistics for numerical columns
        print("\nSummary Statistics:")
        print(df.describe())

        # Extracting specific metrics for summary
        summary = {
            'Total Epochs': df['epoch'].nunique(),
            'Final Train Box Loss': df['train/box_loss'].iloc[-1],
            'Final Train Class Loss': df['train/cls_loss'].iloc[-1],
            'Final Train DFL Loss': df['train/dfl_loss'].iloc[-1],
            'Final Validation Box Loss': df['val/box_loss'].iloc[-1],
            'Final Validation Class Loss': df['val/cls_loss'].iloc[-1],
            'Final Validation DFL Loss': df['val/dfl_loss'].iloc[-1],
            'Final Precision': df['metrics/precision(B)'].iloc[-1],
            'Final Recall': df['metrics/recall(B)'].iloc[-1],
            'Final mAP@50': df['metrics/mAP50(B)'].iloc[-1],
            'Final mAP@50-95': df['metrics/mAP50-95(B)'].iloc[-1]
        }

        print("\nSummary of Metrics:")
        for metric, value in summary.items():
            print(f"{metric}: {value}")
        
except FileNotFoundError:
    print(f"File not found: {csv_file_path}")
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
