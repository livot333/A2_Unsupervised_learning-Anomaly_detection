import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler


class OneClassSVMModel:
    """
    One-Class SVM for anomaly detection.

    Learns a decision boundary around normal training data in a high-dimensional
    kernel space. Points outside the boundary are classified as anomalies.
    Best suited for low-to-medium dimensional data; can be slow on large datasets.
    """

    def __init__(self, training_data_dict, testing_data_dict):
        self.train_dict = training_data_dict
        self.test_dict = testing_data_dict
        self.models = {}
        self.scalers = {}
        self.feature_masks = {}

    def fit_all(self, kernel='rbf', nu=0.05, gamma='scale'):
        """
        Fit a separate One-Class SVM per channel on training data.

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid').
            nu:     Upper bound on fraction of training errors and lower bound on
                    support vectors — acts as a contamination estimate (0 < nu <= 1).
            gamma:  Kernel coefficient ('scale', 'auto', or float).
        """
        print(f"--- Starting Batch One-Class SVM Fit ({len(self.train_dict)} channels) ---")

        for cid, data in self.train_dict.items():
            df = pd.DataFrame(data)
            non_constant = df.columns[df.nunique() > 1].tolist()
            self.feature_masks[cid] = non_constant

            if len(non_constant) == 0:
                print(f"  Channel {cid:6}: skipped (all features constant)")
                continue

            scaler = RobustScaler()
            scaled = scaler.fit_transform(data[:, non_constant])

            svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
            svm.fit(scaled)

            self.models[cid] = svm
            self.scalers[cid] = scaler

    def get_batch_predictions(self, threshold_percentile=None):
        """
        Score test data and return anomaly indices.

        If threshold_percentile is set, uses the training score distribution to set
        a percentile-based threshold (overrides the nu-based boundary).

        Returns:
            predictions_dict: { 'chan_id': [list_of_outlier_indices] }
        """
        predictions_dict = {}

        for cid, svm in self.models.items():
            if cid not in self.test_dict:
                continue

            mask = self.feature_masks[cid]
            scaler = self.scalers[cid]

            scaled_test = scaler.transform(self.test_dict[cid][:, mask])

            if threshold_percentile is not None:
                scaled_train = scaler.transform(self.train_dict[cid][:, mask])
                train_scores = svm.score_samples(scaled_train)
                threshold = np.percentile(train_scores, threshold_percentile)
                test_scores = svm.score_samples(scaled_test)
                outlier_indices = np.where(test_scores <= threshold)[0]
            else:
                # predict returns -1 for anomalies, +1 for normal
                preds = svm.predict(scaled_test)
                outlier_indices = np.where(preds == -1)[0]

            predictions_dict[cid] = outlier_indices.tolist()

        return predictions_dict
