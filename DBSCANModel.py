import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


class DBSCANModel:
    """
    DBSCAN used as a clustering-based anomaly detector.

    DBSCAN naturally identifies outliers as noise points (label = -1) —
    points that do not belong to any dense cluster. No threshold tuning
    is required; the eps and min_samples parameters control sensitivity.
    """

    def __init__(self, training_data_dict, testing_data_dict):
        self.train_dict = training_data_dict
        self.test_dict = testing_data_dict
        self.models = {}
        self.scalers = {}
        self.feature_masks = {}

    def k_distance_plot(self, channel_id, k=5):
        """Plot sorted k-NN distances to help choose eps (look for the 'knee')."""
        data = self.train_dict[channel_id]
        df = pd.DataFrame(data)
        non_constant = df.columns[df.nunique() > 1].tolist()
        scaler = RobustScaler()
        scaled = scaler.fit_transform(data[:, non_constant])

        nbrs = NearestNeighbors(n_neighbors=k).fit(scaled)
        distances, _ = nbrs.kneighbors(scaled)
        k_distances = np.sort(distances[:, -1])[::-1]

        plt.figure(figsize=(8, 4))
        plt.plot(k_distances)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-NN distance')
        plt.title(f'K-Distance Plot (k={k}) — {channel_id}')
        plt.tight_layout()
        plt.show()

    def k_distance_plot_all(self, k=5):
        """Plot sorted k-NN distances for all channels in a normalised single plot."""
        channels = sorted(self.train_dict.keys())
        colors = plt.cm.tab20.colors

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, cid in enumerate(channels):
            data = self.train_dict[cid]
            df = pd.DataFrame(data)
            non_constant = df.columns[df.nunique() > 1].tolist()
            scaler = RobustScaler()
            scaled = scaler.fit_transform(data[:, non_constant])

            nbrs = NearestNeighbors(n_neighbors=k).fit(scaled)
            distances, _ = nbrs.kneighbors(scaled)
            k_distances = np.sort(distances[:, -1])[::-1]

            # Normalise x to 0-1 so channels with different lengths are comparable
            x = np.linspace(0, 1, len(k_distances))

            ax.plot(x, k_distances, color=colors[i % len(colors)], label=cid, linewidth=1.2)

        ax.axhline(y=2.0, color='black', linestyle='--', linewidth=1.2, label='eps=2.0')
        ax.set_ylim(0, 10)
        ax.set_xlabel('Points sorted by distance (normalised)')
        ax.set_ylabel(f'{k}-NN Distance (raw)')
        ax.set_title(f'K-Distance Plot (k={k}) — All Channels (Raw Distances, y capped at 10)')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def fit_all(self, eps=1.5, min_samples=10):
        """
        Fit DBSCAN per channel on training data.
        Noise points (label = -1) are treated as anomalies.
        """
        print(f"--- Starting Batch DBSCAN Fit ({len(self.train_dict)} channels) ---")

        for cid, data in self.train_dict.items():
            df = pd.DataFrame(data)
            non_constant = df.columns[df.nunique() > 1].tolist()
            self.feature_masks[cid] = non_constant

            if len(non_constant) == 0:
                print(f"  Channel {cid:6}: skipped (all features constant)")
                continue

            scaler = RobustScaler()
            scaled = scaler.fit_transform(data[:, non_constant])

            db = DBSCAN(eps=eps, min_samples=min_samples)
            db.fit(scaled)

            n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            n_noise = np.sum(db.labels_ == -1)

            self.models[cid] = db
            self.scalers[cid] = scaler

            print(f"  Channel {cid:6}: {n_clusters} clusters  |  {n_noise} noise points "
                  f"({100 * n_noise / len(scaled):.1f}%)")

    def get_batch_predictions(self, eps=None, min_samples=None):
        """
        Predict anomalies on test data by re-running DBSCAN.
        Noise points (label = -1) are returned as outlier indices.

        Returns:
            predictions_dict: { 'chan_id': [list_of_outlier_indices] }
        """
        predictions_dict = {}

        for cid, db in self.models.items():
            if cid not in self.test_dict:
                continue

            mask = self.feature_masks[cid]
            scaler = self.scalers[cid]
            scaled_test = scaler.transform(self.test_dict[cid][:, mask])

            # Use same eps/min_samples as training unless overridden
            _eps = eps if eps is not None else db.eps
            _min_samples = min_samples if min_samples is not None else db.min_samples

            db_test = DBSCAN(eps=_eps, min_samples=_min_samples)
            labels = db_test.fit_predict(scaled_test)

            outlier_indices = np.where(labels == -1)[0]
            predictions_dict[cid] = outlier_indices.tolist()

        return predictions_dict
