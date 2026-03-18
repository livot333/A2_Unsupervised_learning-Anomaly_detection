import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


class KMeansAnalyzer:
    """
    K-Means used as a clustering and preprocessing tool.

    Step 1 - Clustering: identify natural operating mode clusters in training data.
    Step 2 - Preprocessing: append centroid distances to the feature set, enriching
             the input for downstream anomaly detection models (e.g. IsolationForest).
    """

    def __init__(self, training_data_dict, testing_data_dict):
        self.train_dict = training_data_dict
        self.test_dict = testing_data_dict
        self.models = {}
        self.scalers = {}
        self.feature_masks = {}
        self.best_k = {}

    def elbow_plot(self, channel_id, k_range=range(2, 15), random_seed=42):
        """Plot inertia vs K to help choose the number of clusters for a channel."""
        data = self.train_dict[channel_id]
        df = pd.DataFrame(data)
        non_constant = df.columns[df.nunique() > 1].tolist()
        scaler = RobustScaler()
        scaled = scaler.fit_transform(data[:, non_constant])

        inertia = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_seed, n_init='auto')
            km.fit(scaled)
            inertia.append(km.inertia_)

        plt.figure(figsize=(8, 4))
        plt.plot(list(k_range), inertia, marker='o')
        plt.xlabel('Number of clusters K')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Method — K-Means ({channel_id})')
        plt.tight_layout()
        plt.show()

    def fit_all(self, k=5, random_seed=42):
        """Fit K-Means per channel for clustering analysis."""
        print(f"--- Starting Batch K-Means Fit ({len(self.train_dict)} channels) ---")
        for cid, data in self.train_dict.items():
            df = pd.DataFrame(data)
            non_constant = df.columns[df.nunique() > 1].tolist()
            self.feature_masks[cid] = non_constant

            if len(non_constant) == 0:
                print(f"  Channel {cid:6}: skipped (all features constant)")
                continue

            scaler = RobustScaler()
            scaled = scaler.fit_transform(data[:, non_constant])

            km = KMeans(n_clusters=k, random_state=random_seed, n_init='auto')
            km.fit(scaled)

            self.models[cid] = km
            self.scalers[cid] = scaler
            self.best_k[cid] = k
            print(f"  Channel {cid:6}: K-Means fitted  K={k}  |  cluster sizes: {np.bincount(km.labels_)}")

    def plot_clusters(self, channel_id, X_2d, title=None):
        """Visualise cluster assignments in 2D PCA space."""
        km = self.models[channel_id]
        mask = self.feature_masks[channel_id]
        scaler = self.scalers[channel_id]
        scaled = scaler.transform(self.train_dict[channel_id][:, mask])
        labels = km.predict(scaled)

        title = title or f'K-Means Clusters (K={self.best_k[channel_id]}) — {channel_id}'
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=2, alpha=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.title(title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.show()

    def get_enriched_features(self):
        """
        Append centroid distances to original features.
        Returns enriched train and test dicts for downstream anomaly detectors.
        """
        train_out, test_out = {}, {}
        for cid, km in self.models.items():
            mask = self.feature_masks[cid]
            scaler = self.scalers[cid]

            train_scaled = scaler.transform(self.train_dict[cid][:, mask])
            test_scaled  = scaler.transform(self.test_dict[cid][:, mask])

            train_out[cid] = np.hstack([self.train_dict[cid], km.transform(train_scaled)])
            test_out[cid]  = np.hstack([self.test_dict[cid],  km.transform(test_scaled)])

            print(f"  Channel {cid}: {self.train_dict[cid].shape[1]} → {train_out[cid].shape[1]} features")
        return train_out, test_out

    def get_batch_predictions(self, threshold_percentile=95):
        """
        Use centroid distance as anomaly score.
        Returns: { chan_id: [outlier_indices] }
        """
        predictions_dict = {}
        for cid, km in self.models.items():
            if cid in self.test_dict:
                mask = self.feature_masks[cid]
                scaler = self.scalers[cid]

                train_scaled = scaler.transform(self.train_dict[cid][:, mask])
                test_scaled  = scaler.transform(self.test_dict[cid][:, mask])

                train_scores = km.transform(train_scaled).min(axis=1)
                test_scores  = km.transform(test_scaled).min(axis=1)

                threshold = np.percentile(train_scores, threshold_percentile)
                outlier_indices = np.where(test_scores > threshold)[0]
                predictions_dict[cid] = outlier_indices.tolist()

        return predictions_dict
