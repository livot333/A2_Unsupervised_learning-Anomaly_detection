import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sk_PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
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
            # scaler = RobustScaler()
            pca = sk_PCA(n_components=n_components)
            
            scaled_data = scaler.fit_transform(data)
            pca.fit(scaled_data)
            
            # 2. Store the objects
            self.models[cid] = pca
            self.scalers[cid] = scaler
            
            var_exp = np.sum(pca.explained_variance_ratio_) * 100
            print(f" Channel {cid:6}: Explained Variance {var_exp:6.2f}%")

    def get_PCA_predictions(self, mode="test", threshold_percentile=98):
        """
        Calculates reconstruction errors and converts them directly into 
        outlier indices for the evaluation structure.
        
        Returns:
            predictions_dict: { 'chan_id': [list_of_outlier_indices] }
        """
        data_source = self.test_dict if mode == "test" else self.train_dict
        predictions_dict = {}

        for cid, pca in self.models.items():
            if cid in data_source:
                data = data_source[cid]
                scaler = self.scalers[cid]
                
                # 1. Standard reconstruction process
                scaled_data = scaler.transform(data)
                reduced = pca.transform(scaled_data)
                reconstructed_scaled = pca.inverse_transform(reduced)
                
                # 2. Calculate MSE per row
                mse = np.mean(np.power(scaled_data - reconstructed_scaled, 2), axis=1)
                
                # 3. Apply threshold to get indices
                # We calculate the threshold dynamically for each channel
                current_threshold = np.percentile(mse, threshold_percentile)
                
                # Find positions where error is above threshold
                outlier_indices = np.where(mse >= current_threshold)[0]
                
                # 4. Store as a list of indices
                predictions_dict[cid] = outlier_indices.tolist()
                
        return predictions_dict

    def transform_PCA(self, mode="test"):
        """returns data previouslz transformed from PCA (n_components)."""
        data_source = self.test_dict if mode == "test" else self.train_dict
        transformed_dict = {}

        for cid, pca in self.models.items():
            if cid in data_source:
                scaled_data = self.scalers[cid].transform(data_source[cid])
                # Tady je to kouzlo: bereme jen ty hlavní komponenty
                transformed_dict[cid] = pca.transform(scaled_data)
        
        return transformed_dict