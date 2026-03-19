import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class IsolationForestModel:
    """
    Isolation Forest for anomaly detection.

    Isolates anomalies by randomly partitioning the feature space —
    anomalous points require fewer splits to isolate (shorter path length).
    Well-suited for high-dimensional data and does not assume any data distribution.
    """

    def __init__(self, training_data_dict, testing_data_dict):
        self.train_dict = training_data_dict
        self.test_dict = testing_data_dict
        self.models = {}
        self.scalers = {}
        self.feature_masks = {}

    def fit_all(self, contamination=0.05, n_estimators=100, random_state=42):
        """
        Fit a separate Isolation Forest per channel on training data.

        Args:
            contamination: Expected fraction of anomalies (domain estimate, not from labels).
            n_estimators:  Number of isolation trees.
            random_state:  Reproducibility seed.
        """
        print(f"--- Starting Batch Isolation Forest Fit ({len(self.train_dict)} channels) ---")

        for cid, data in self.train_dict.items():
            df = pd.DataFrame(data)
            non_constant = df.columns[df.nunique() > 1].tolist()
            self.feature_masks[cid] = non_constant

            if len(non_constant) == 0:
                print(f"  Channel {cid:6}: skipped (all features constant)")
                continue

            scaler = RobustScaler()
            scaled = scaler.fit_transform(data[:, non_constant])

            iso = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1
            )
            iso.fit(scaled)

            self.models[cid] = iso
            self.scalers[cid] = scaler


    def get_batch_predictions(self, threshold_percentile=None):
        """
        Score test data and return anomaly indices.

        If threshold_percentile is set, uses the training score distribution to set
        a percentile-based threshold (overrides the contamination-based boundary).

        Returns:
            predictions_dict: { 'chan_id': [list_of_outlier_indices] }
        """
        predictions_dict = {}

        for cid, iso in self.models.items():
            if cid not in self.test_dict:
                continue

            mask = self.feature_masks[cid]
            scaler = self.scalers[cid]

            scaled_test = scaler.transform(self.test_dict[cid][:, mask])

            if threshold_percentile is not None:
                # Score-based threshold from training distribution
                scaled_train = scaler.transform(self.train_dict[cid][:, mask])
                train_scores = iso.score_samples(scaled_train)
                threshold = np.percentile(train_scores, threshold_percentile)
                test_scores = iso.score_samples(scaled_test)
                outlier_indices = np.where(test_scores <= threshold)[0]
            else:
                # Use contamination-based boundary (predict returns -1 for anomalies)
                preds = iso.predict(scaled_test)
                outlier_indices = np.where(preds == -1)[0]

            predictions_dict[cid] = outlier_indices.tolist()

        return predictions_dict
