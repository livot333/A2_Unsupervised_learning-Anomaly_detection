import numpy as np
import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Ignorujeme varování o nedoběhnutí iterací během tuningu (pro zrychlení)
warnings.filterwarnings("ignore", category=UserWarning)

class MultiFileSklearnTuner:
    def __init__(self, target_cids=None, n_trials=10):
        """
        Args:
            target_cids: Seznam jmen souborů (např. ['T-9', 'D-12']) nebo None.
            n_trials: Počet pokusů pro každý soubor (čím více, tím lepší F1).
        """
        # Převedeme vstup na seznam, ať už uživatel zadá string nebo list
        if target_cids is None:
            self.target_cids = None
        elif isinstance(target_cids, list):
            self.target_cids = target_cids
        else:
            self.target_cids = [target_cids]

        self.n_trials = n_trials
        self.best_params_per_file = {}

    def _objective(self, trial, cid, train_data, test_data, evaluation_obj):
        # Parametry, které Optuna ladí pro každý soubor zvlášť
        latent_dim = trial.suggest_int("latent_dim", 2, 12)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        percentile = trial.suggest_float("percentile", 90.0, 99.9)

        # Standardizace (vždy fit na train, transform na test)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # MLP jako Autoencoder
        model = MLPRegressor(
            hidden_layer_sizes=(latent_dim,),
            alpha=alpha,
            max_iter=80, # Méně pro tuning, víc pro finále
            random_state=42
        )
        
        # Učení (rekonstrukce vstupu)
        model.fit(train_scaled, train_scaled)
        reconstructed = model.predict(test_scaled)
        
        # Výpočet chyby (MSE)
        mse = np.mean((test_scaled - reconstructed)**2, axis=1)
        
        # Predikce na základě navrženého percentilu
        thresh = np.percentile(mse, percentile)
        preds = {cid: set(np.where(mse > thresh)[0])}

        # Vyhodnocení proti NASA labelům (hledáme maximum F1)
        report = evaluation_obj.compare_methods_results(preds)
        f1 = report.loc[0, 'F1_Score'] if not report.empty else 0
        return f1

    def tune_and_predict(self, train_dict, test_dict, evaluation_obj):
        """
        Projde zadané soubory, pro každý najde nejlepší parametry a vrátí finální predikce.
        """
        # Pokud uživatel nezadal nic, projde vše, co je v dictionary
        ids_to_process = self.target_cids if self.target_cids else list(train_dict.keys())
        all_final_outliers = {}

        print(f"--- Zahajuji trénování pro {len(ids_to_process)} souborů ---")

        for cid in ids_to_process:
            if cid not in train_dict or cid not in test_dict:
                print(f"Skipping {cid}: Soubor nebyl nalezen v datech.")
                continue

            print(f"\n[SOUBOR {cid}] - Startuju optimalizaci (n_trials={self.n_trials})")
            
            # 1. Hledání nejlepších parametrů
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(trial, cid, train_dict[cid], test_dict[cid], evaluation_obj), 
                n_trials=self.n_trials,
                show_progress_bar=False
            )
            
            best = study.best_params
            self.best_params_per_file[cid] = best
            print(f"-> Nejlepší F1 pro {cid}: {study.best_value:.4f} (Latent: {best['latent_dim']}, P: {best['percentile']:.2f})")

            # 2. Finální trénink s nejlepšími parametry
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_dict[cid])
            test_scaled = scaler.transform(test_dict[cid])

            final_model = MLPRegressor(
                hidden_layer_sizes=(best['latent_dim'],),
                alpha=best['alpha'],
                max_iter=250, # Teď mu dáme čas se to naučit pořádně
                random_state=42
            )
            final_model.fit(train_scaled, train_scaled)
            final_reconstructed = final_model.predict(test_scaled)
            final_mse = np.mean((test_scaled - final_reconstructed)**2, axis=1)
            
            # Finální set outlierů
            thresh = np.percentile(final_mse, best['percentile'])
            all_final_outliers[cid] = set(np.where(final_mse > thresh)[0])

        return all_final_outliers