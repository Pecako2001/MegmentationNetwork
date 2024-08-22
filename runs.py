import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the main directory containing the training folders
main_directory = 'detect'

# Initialize a dictionary to hold data from all folders
training_data = {}

# Loop through each folder in the main directory
for folder in os.listdir(main_directory):
    print(f'Processing folder: {folder}')
    folder_path = os.path.join(main_directory, folder)
    if os.path.isdir(folder_path):
        csv_path = os.path.join(folder_path, 'results.csv')
        if os.path.exists(csv_path):
            try:
                # Read the CSV file and strip leading/trailing whitespace from column names
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()  # Strip whitespace from headers

                # Check for required columns and store the data
                if all(col in df.columns for col in ['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']):
                    training_data[folder] = df[['epoch', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']]
                else:
                    print(f"Required columns not found in {csv_path}")
            except Exception as e:
                print(f"Error reading file: {csv_path} - {e}")

# Plotting precision, recall, mAP50, and mAP50-95 over epochs
metrics = {
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': 'mAP50',
    'metrics/mAP50-95(B)': 'mAP50-95'
}

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()  # Flatten the 2D array of axes to make iteration easier

# Plot each metric in a separate subplot
for i, (metric, metric_name) in enumerate(metrics.items()):
    for folder_name, df in training_data.items():
        if metric in df.columns:
            axs[i].plot(df['epoch'], df[metric], label=folder_name)
    
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel(metric_name)
    axs[i].set_title(f'{metric_name} Over Epochs')
    axs[i].legend()
    axs[i].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
