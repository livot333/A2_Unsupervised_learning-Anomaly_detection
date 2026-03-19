import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

class GMM:
    def __init__(self, training_data_dict, testing_data_dict):
        self.train_dict = training_data_dict
        self.test_dict = testing_data_dict
        self.models = {} 
        self.scalers = {}
        self.feature_masks = {} # Store indices of non-constant columns

    def fit_all(self, n_components=1,covariance_type='full',n_init=5, max_iter=100, tol=1e-3, init_params='kmeans', warm_start=True, reg_covar=1e-3):
        """
        Fits a separate GMM for each channel. 
        n_components=1 is often best for anomaly detection (modeling the 'normal' cluster).
        """
        print(f"--- Starting Batch GMM Fit ({len(self.train_dict)} channels) ---")
        
        for cid, data in self.train_dict.items():
            # 1. Feature selection (Remove constant columns as we discussed)
            df = pd.DataFrame(data)
            non_constant_indices = df.columns[df.nunique() > 1].tolist()
            self.feature_masks[cid] = non_constant_indices
            
            filtered_train = data[:, non_constant_indices]
            
            # 2. Scaling
            scaler = RobustScaler()
            scaled_train = scaler.fit_transform(filtered_train)
            
            # 3. Fit GMM
            # covariance_type='full' allows for complex shapes of normal data
            gmm = GaussianMixture(
                n_components=n_components, 
                n_init=n_init,           # Added: prevents local minima
                max_iter=max_iter,       # Added: limits computation time
                tol=tol,                 # Added: defines when model is "finished"
                covariance_type=covariance_type, 
                random_state=42,
                warm_start=warm_start,
                init_params=init_params,
                reg_covar=reg_covar
            )

            
            gmm.fit(scaled_train)
            
            self.models[cid] = gmm
            self.scalers[cid] = scaler
            
            

    def get_batch_predictions(self, threshold_percentile=5):
        """
        Calculates log-likelihood scores and flags lowest scores as anomalies.
        
        Returns:
            predictions_dict: { 'chan_id': [list_of_outlier_indices] }
        """
        predictions_dict = {}

        for cid, gmm in self.models.items():
            if cid in self.test_dict:
                # 1. Alignment with training features
                mask = self.feature_masks[cid]
                data = self.test_dict[cid][:, mask]
                
                # 2. Transform test data
                scaler = self.scalers[cid]
                scaled_test = scaler.transform(data)
                
                # 3. Calculate Log-Likelihood (Score)
                # Lower score means the point is less likely to belong to the normal distribution
                scores = gmm.score_samples(scaled_test)
                
                # 4. Thresholding
                # In GMM, anomalies are in the LOWEST percentile of scores
                threshold = np.percentile(scores, threshold_percentile)
                outlier_indices = np.where(scores <= threshold)[0]
                
                predictions_dict[cid] = outlier_indices.tolist()
                
        return predictions_dict