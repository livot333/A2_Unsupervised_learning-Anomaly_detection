import numpy as np
import pandas as pd
import os

def inspect_dataset(data_dir):
    """
    Scans the directory for .npy files and returns a summary of their properties.
    """
    print(f"--- Inspecting Directory: {data_dir} ---")
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    
    if not files:
        print("No .npy files found.")
        return None

    report = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path)
        
        # Checking for missing values
        nan_count = np.isnan(data).sum()
        
        report.append({
            'Channel ID': file.replace('.npy', ''),
            'Rows (Timesteps)': data.shape[0],
            'Cols (Features)': data.shape[1],
            'Missing Values': nan_count
        })
    
    return pd.DataFrame(report)

# Define paths
train_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\train"
test_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\test"

# Run inspection
train_info = inspect_dataset(train_path)
test_info = inspect_dataset(test_path)

print("\nTrain Dataset Summary (First 5):")
print(train_info.head())

print("\nTest Dataset Summary (First 5):")
print(test_info.head())