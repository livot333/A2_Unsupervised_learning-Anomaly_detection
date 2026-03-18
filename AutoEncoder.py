import numpy as np
import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore warnings about non-convergence during tuning iterations to keep output clean and fast
warnings.filterwarnings("ignore", category=UserWarning)

class MultiFileSklearnTuner:
    def __init__(self, target_cids=None, n_trials=10):
        """
        Args:
            target_cids: List of file names (e.g., ['T-9', 'D-12']) or None to process all.
            n_trials: Number of trials per file (higher count usually leads to better F1).
        """
        # Convert input to a list regardless of whether a string or list is provided
        if target_cids is None:
            self.target_cids = None
        elif isinstance(target_cids, list):
            self.target_cids = target_cids
        else:
            self.target_cids = [target_cids]

        self.n_trials = n_trials
        self.best_params_per_file = {}

    def _objective(self, trial, cid, train_data, test_data, evaluation_obj):
        # Hyperparameters tuned by Optuna specifically for each individual file
        latent_dim = trial.suggest_int("latent_dim", 2, 12)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        percentile = trial.suggest_float("percentile", 90.0, 99.9)

        # Standardization (always fit on train, transform on test)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Using MLP as an Autoencoder (bottleneck architecture)
        model = MLPRegressor(
            hidden_layer_sizes=(latent_dim,),
            alpha=alpha,
            max_iter=80, # Lower iterations for faster tuning
            random_state=42
        )
        
        # Training (input reconstruction: X = y)
        model.fit(train_scaled, train_scaled)
        reconstructed = model.predict(test_scaled)
        
        # Reconstruction Error calculation (MSE)
        mse = np.mean((test_scaled - reconstructed)**2, axis=1)
        
        # Prediction based on the suggested percentile threshold
        thresh = np.percentile(mse, percentile)
        preds = {cid: set(np.where(mse > thresh)[0])}

        # Evaluation against NASA labels (maximizing the F1 Score)
        report = evaluation_obj.compare_methods_results(preds)
        f1 = report.loc[0, 'F1_Score'] if not report.empty else 0
        return f1

    def tune_and_predict(self, train_dict, test_dict, evaluation_obj):
        """
        Iterates through specified files, finds the best parameters for each, and returns final predictions.
        """
        # If no target_cids specified, process all keys in the training dictionary
        ids_to_process = self.target_cids if self.target_cids else list(train_dict.keys())
        all_final_outliers = {}

        print(f"--- Starting training for {len(ids_to_process)} files ---")

        for cid in ids_to_process:
            if cid not in train_dict or cid not in test_dict:
                print(f"Skipping {cid}: File not found in dataset.")
                continue

            print(f"\n[FILE {cid}] - Starting optimization (n_trials={self.n_trials})")
            
            # 1. Search for optimal hyperparameters
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(trial, cid, train_dict[cid], test_dict[cid], evaluation_obj), 
                n_trials=self.n_trials,
                show_progress_bar=False
            )
            
            best = study.best_params
            self.best_params_per_file[cid] = best
            print(f"-> Best F1 for {cid}: {study.best_value:.4f} (Latent: {best['latent_dim']}, P: {best['percentile']:.2f})")

            # 2. Final training using the best discovered parameters
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_dict[cid])
            test_scaled = scaler.transform(test_dict[cid])

            final_model = MLPRegressor(
                hidden_layer_sizes=(best['latent_dim'],),
                alpha=best['alpha'],
                max_iter=250, # Increased iterations for the final model fit
                random_state=42
            )
            final_model.fit(train_scaled, train_scaled)
            final_reconstructed = final_model.predict(test_scaled)
            final_mse = np.mean((test_scaled - final_reconstructed)**2, axis=1)
            
            # Final outlier set generation
            thresh = np.percentile(final_mse, best['percentile'])
            all_final_outliers[cid] = set(np.where(final_mse > thresh)[0])

        return all_final_outliers