import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sk_PCA
from sklearn.preprocessing import StandardScaler
import ast

class BatchPCA:
    def __init__(self, training_data_dict, testing_data_dict):
        self.train_dict = training_data_dict
        self.test_dict = testing_data_dict
        # Dictionary to store individual PCA models and scalers for each channel
        self.models = {} 
        self.scalers = {}

    def fit_all(self, channel_ids=None, n_components=3):
        """
        Fits a separate PCA model for each channel in the list.
        If channel_ids is None, it fits all available channels.
        """
        target_channels = channel_ids if channel_ids else list(self.train_dict.keys())
        
        print(f"--- Starting Batch PCA Fit ({len(target_channels)} channels) ---")
        
        for cid in target_channels:
            if cid not in self.train_dict:
                continue
            
            data = self.train_dict[cid]
            
            # 1. Scale and Fit
            scaler = StandardScaler()
            pca = sk_PCA(n_components=n_components)
            
            scaled_data = scaler.fit_transform(data)
            pca.fit(scaled_data)
            
            # 2. Store the objects
            self.models[cid] = pca
            self.scalers[cid] = scaler
            
            var_exp = np.sum(pca.explained_variance_ratio_) * 100
            print(f" Channel {cid:6}: Explained Variance {var_exp:6.2f}%")

    def get_batch_reconstruction_errors(self, mode="test"):
        """
        Calculates reconstruction errors for all fitted channels.
        Returns a dictionary: {channel_id: error_array}
        """
        data_source = self.test_dict if mode == "test" else self.train_dict
        all_errors = {}

        for cid, pca in self.models.items():
            if cid in data_source:
                data = data_source[cid]
                scaler = self.scalers[cid]
                
                # Transform -> Inverse Transform
                scaled_data = scaler.transform(data)
                reduced = pca.transform(scaled_data)
                reconstructed_scaled = pca.inverse_transform(reduced)
                
                # Calculate MSE (Mean Squared Error) per row
                mse = np.mean(np.power(scaled_data - reconstructed_scaled, 2), axis=1)
                all_errors[cid] = mse
                
        return all_errors

    def plot_summary(self, errors_dict):
        """
        Visualizes the average error per channel to quickly spot problematic sensors.
        """
        avg_errors = {cid: np.mean(err) for cid, err in errors_dict.items()}
        
        plt.figure(figsize=(12, 5))
        plt.bar(avg_errors.keys(), avg_errors.values(), color='salmon')
        plt.xticks(rotation=45)
        plt.ylabel("Mean Reconstruction Error")
        plt.title("Average Anomaly Score across Channels")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


    def evaluate_pca_anomalies(self,errors_dict, labels_df, threshold_percentile=None):
        """
        Compares PCA reconstruction errors with NASA ground truth labels.
        
        Args:
            errors_dict: Dictionary {channel_id: np.array of errors} from PCA.
            labels_df: The DataFrame you shared (containing 'chan_id' and 'anomaly_sequences').
            threshold_percentile: Top X% of errors to be flagged as anomalies (default 5%).
        """
        results = []

        for chan_id, errors in errors_dict.items():
            # 1. Get ground truth for this channel
            label_row = labels_df[labels_df['chan_id'] == chan_id]
            if label_row.empty:
                continue
                
            # Get total length and ground truth ranges
            num_values = label_row.iloc[0]['num_values']
            
            # Safe conversion of string "[[start, end], ...]" to Python list
            sequences = label_row.iloc[0]['anomaly_sequences']
            if isinstance(sequences, str):
                sequences = ast.literal_eval(sequences)
                
            # Create Ground Truth Mask (array of 0s and 1s)
            y_true = np.zeros(num_values)
            for start, end in sequences:
                y_true[start : end + 1] = 1 # inclusive range
                
            # 2. Create Prediction Mask using a threshold
            # We flag the top X% of highest errors as anomalies
            threshold = np.percentile(errors, threshold_percentile)
            y_pred = (errors >= threshold).astype(int)
            
            # Align lengths (NASA labels are for the full test file)
            # Ensure we compare only the overlapping part
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # 3. Calculate metrics
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'Channel': chan_id,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1_Score': round(f1, 4),
                'True_Anom_Points': int(np.sum(y_true)),
                'Detected_Points': int(np.sum(y_pred))
            })
            
        return pd.DataFrame(results)