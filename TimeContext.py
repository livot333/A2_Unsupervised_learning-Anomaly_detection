import numpy as np
import pandas as pd

class TimeContextModif:
    def __init__(self, train_dataset, test_dataset):
        """
        Args:
            train_dataset: Dictionary {chan_id: np.array}
            test_dataset: Dictionary {chan_id: np.array}
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def apply_sliding_window(self, window_length=10, flatten=True):
        """
        Transforms data into overlapping windows.
        If flatten=True: (N, features) -> (N - L + 1, features * L)
        If flatten=False: (N, features) -> (N - L + 1, L, features) -- for LSTM
        """
        new_train = {}
        new_test = {}

        for datasets, target_dict in [(self.train_dataset, new_train), 
                                      (self.test_dataset, new_test)]:
            for cid, data in datasets.items():
                n_samples, n_features = data.shape
                if n_samples < window_length:
                    continue
                
                # Create windows using advanced numpy indexing (stride_tricks) for speed
                shape = (n_samples - window_length + 1, window_length, n_features)
                strides = (data.strides[0], data.strides[0], data.strides[1])
                windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
                
                if flatten:
                    # Flatten window depth: (N, L, F) -> (N, L*F)
                    target_dict[cid] = windows.reshape(windows.shape[0], -1)
                else:
                    target_dict[cid] = windows
                    
        return new_train, new_test

    def add_lag_features(self, lags=[1, 2, 3]):
        """
        Adds shifted versions of the features to the current row.
        (N, features) -> (N - max(lags), features * (1 + len(lags)))
        """
        new_train = {}
        new_test = {}

        max_lag = max(lags)

        for datasets, target_dict in [(self.train_dataset, new_train), 
                                      (self.test_dataset, new_test)]:
            for cid, data in datasets.items():
                n_samples, n_features = data.shape
                if n_samples <= max_lag:
                    continue
                
                # Start with the original data (trimmed by max_lag)
                combined = [data[max_lag:]]
                
                # Add lagged versions
                for lag in lags:
                    combined.append(data[max_lag - lag : -lag])
                
                # Horizontal stack: (N, F) + (N, F) + ... -> (N, F * (1+lags))
                target_dict[cid] = np.hstack(combined)
                
        return new_train, new_test
    

    def add_rolling_statistics(self, window_length=500):
        new_train = {}
        new_test = {}

        for datasets, target_dict in [(self.train_dataset, new_train), 
                                      (self.test_dataset, new_test)]:
            for cid, data in datasets.items():
                df = pd.DataFrame(data)
                
                # min_periods=1 zajistí, že dostaneme výsledek i pro začátek souboru
                # (okno se postupně "naplňuje")
                roll = df.rolling(window=window_length, min_periods=1)
                
                r_mean = roll.mean()
                r_std = roll.std().fillna(0) # Std u jednoho vzorku je NaN, nahradíme 0
                r_min = roll.min()
                r_max = roll.max()
                
                combined = pd.concat([df, r_mean, r_std, r_min, r_max], axis=1)
                
                # Místo uříznutí natvrdo teď vezmeme všechna data. 
                # Tím pádem délka test_win bude stejná jako u test_data_dict.
                target_dict[cid] = combined.values
                
        return new_train, new_test
    
    def add_spectral_features(self, window_length=500):
        """
        Extracts spectral energy from rolling windows using FFT.
        This helps detect frequency-based contextual anomalies.
        """
        new_train = {}
        new_test = {}

        for datasets, target_dict in [(self.train_dataset, new_train), 
                                      (self.test_dataset, new_test)]:
            for cid, data in datasets.items():
                num_samples, num_sensors = data.shape
                # Připravíme pole pro výslednou "energii" (stejný rozměr jako data)
                spectral_energy = np.zeros((num_samples, num_sensors))

                for s in range(num_sensors):
                    sensor_data = data[:, s]
                    # Rolling FFT energy calculation
                    for i in range(window_length, num_samples):
                        window = sensor_data[i - window_length : i]
                        # FFT a výpočet magnitudy (energie)
                        fft_vals = np.fft.rfft(window)
                        energy = np.sum(np.abs(fft_vals)**2) / window_length
                        spectral_energy[i, s] = energy
                
                # Spojíme původní data se spektrální energií
                combined = np.hstack([data, spectral_energy])
                target_dict[cid] = combined
                
        return new_train, new_test
    

    def add_derivative_features(self):
        """
        Calculates Velocity (1st derivative) and Acceleration (2nd derivative).
        Captures sudden jerks and rate-of-change anomalies.
        """
        new_train = {}
        new_test = {}


        for datasets, target_dict in [(self.train_dataset, new_train), 
                                      (self.test_dataset, new_test)]:
            for cid, data in datasets.items():
                # num_samples x num_sensors
                
                # 1. Rychlost (Velocity) - první derivace
                # np.gradient počítá centrální diferenci, zachovává tvar (shape)
                velocity = np.gradient(data, axis=0)
                
                # 2. Zrychlení (Acceleration) - druhá derivace
                acceleration = np.gradient(velocity, axis=0)
                
                # Spojíme: Původní data + Rychlost + Zrychlení
                # Výsledek bude mít 3x více sloupců než původní data
                combined = np.hstack([data, velocity, acceleration])
                
                target_dict[cid] = combined
                
        return new_train, new_test