import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

class LOF:
    def __init__(self, training_data_dict, testing_data_dict):
        self.train_dict = training_data_dict
        self.test_dict = testing_data_dict
        self.models = {} 
        self.scalers = {}
        self.feature_masks = {}

    def fit_all(self, n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, contamination='auto'):
        """
        Fits a separate LOF model for each channel.
        
        Args:
            n_neighbors: Number of neighbors to use (default 20).
            algorithm: Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
            leaf_size: Leaf size passed to BallTree or KDTree (default 30).
            metric: Distance metric to use for the tree (default 'minkowski').
            p: Parameter for the Minkowski metric (p=2 is Euclidean).
            contamination: Expected proportion of outliers in the data (default 'auto').
        """
        print(f"--- Starting Batch LOF Fit ({len(self.train_dict)} channels) ---")
        
        for cid, data in self.train_dict.items():
            # 1. Feature selection (Consistency)
            df = pd.DataFrame(data)
            non_constant_indices = df.columns[df.nunique() > 1].tolist()
            self.feature_masks[cid] = non_constant_indices
            
            filtered_train = data[:, non_constant_indices]
            
            # 2. Scaling
            scaler = RobustScaler()
            scaled_train = scaler.fit_transform(filtered_train)
            
            # 3. Fit LOF
            # novelty=True must be set to use .predict() or .decision_function() on new data
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size,
                metric=metric,
                p=p,
                contamination=contamination,
                novelty=True  # CRITICAL for train/test split
            )
            
            lof.fit(scaled_train)
            
            self.models[cid] = lof
            self.scalers[cid] = scaler
            
          

    def get_batch_predictions(self, threshold_percentile=5):
        """
        Calculates outlier scores and flags lowest scores as anomalies.
        Note: LOF decision_function returns negative values (lower is more anomalous).
        
        Returns:
            predictions_dict: { 'chan_id': [list_of_outlier_indices] }
        """
        predictions_dict = {}

        for cid, lof in self.models.items():
            if cid in self.test_dict:
                mask = self.feature_masks[cid]
                data = self.test_dict[cid][:, mask]
                
                scaler = self.scalers[cid]
                scaled_test = scaler.transform(data)
                
                # decision_function returns the 'offset' of the points.
                # Lower values are more anomalous.
                scores = lof.decision_function(scaled_test)
                
                # Thresholding
                threshold = np.percentile(scores, threshold_percentile)
                outlier_indices = np.where(scores <= threshold)[0]
                
                predictions_dict[cid] = outlier_indices.tolist()
                
        return predictions_dict