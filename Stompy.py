import stumpy
import numpy as np
import pandas as pd

import stumpy
import numpy as np
import pandas as pd

class MstumpDetector:
    def __init__(self, window_size=100):
        """
        Args:
            window_size (int): Délka vzoru (subsequence), kterou porovnáváme.
        """
        self.window_size = window_size
        self.anomaly_scores = {}

    def get_batch_predictions(self, dataset_dict, threshold_percentile=95):
        predictions = {}

        for cid, data in dataset_dict.items():
            n_samples, n_dimensions = data.shape
            m = self.window_size
            
            # Ošetření délky
            if n_samples <= m:
                m = n_samples // 2
            if m < 3: continue

            # ČIŠTĚNÍ: mSTUMP je citlivý na konstantní sloupce
            # Přidáme nepatrný šum, aby std_dev nebyla nikdy 0
            data_cleaned = data.astype(np.float64)
            data_cleaned += np.random.normal(0, 1e-6, data_cleaned.shape)
            
            try:
                # mSTUMP výpočet
                # normalize=False je klíčové, pokud chceme zachovat váhu binárních změn
                # Poznámka: Pokud vaše verze stumpy nepodporuje normalize v mstump, 
                # nechte default, ale nan_to_num to zachrání.
                mp, _ = stumpy.mstump(data_cleaned.T, m=m)
                
                # SOUČET místo posledního sloupce: 
                # Někdy je lepší vzít průměr všech dimenzí, aby jeden senzor nepřehlušil zbytek
                multivariate_profile = np.nan_to_num(mp).mean(axis=0)
                
                # Zarovnání
                pad_width = n_samples - len(multivariate_profile)
                full_scores = np.concatenate([multivariate_profile, np.zeros(pad_width)])
                
                # Pokud jsou všechny skóre 0, threshold nic nenajde
                if np.max(full_scores) == 0:
                    print(f"Channel {cid}: All scores are zero. Check data variance.")
                    predictions[cid] = []
                    continue

                # Snížíme práh na 95 (hledáme 5 % nejhorších), abychom viděli, jestli se něco chytne
                threshold = np.percentile(full_scores, threshold_percentile)
                outlier_indices = np.where(full_scores >= threshold)[0]
                
                predictions[cid] = outlier_indices
                self.anomaly_scores[cid] = full_scores
                
                
            except Exception as e:
                print(f"Error on {cid}: {e}")
                predictions[cid] = []

        return predictions