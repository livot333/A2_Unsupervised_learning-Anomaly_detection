import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import re


class EvaluateResults:
    def __init__(self):
        """
        Initializes the evaluation class without a fixed file path.
        """
        self.solution_dict = {}

    def load_solution(self, labels_df):
        """
        Processes the labels DataFrame and creates a master dictionary of anomaly intervals.
        
        Args:
            labels_df: The DataFrame returned by your dataset loading function.
        """
        if labels_df is None or labels_df.empty:
            print("Error: Labels DataFrame is empty.")
            return None

        for _, row in labels_df.iterrows():
            chan_id = row['chan_id']
            
            # 1. Sequences are usually fine (they are numbers: [[1, 2]])
            sequences = row['anomaly_sequences']
            if isinstance(sequences, str):
                sequences = ast.literal_eval(sequences)
            
            # 2. FIX: Handle the 'class' column strings without quotes
            anomaly_class = row['class']
            if isinstance(anomaly_class, str):
                try:
                    # Regex finds words and wraps them in double quotes
                    # e.g., [point, contextual] -> ["point", "contextual"]
                    sanitized_class = re.sub(r'([a-zA-Z_]\w*)', r'"\1"', anomaly_class)
                    anomaly_class = ast.literal_eval(sanitized_class)
                except Exception:
                    # Fallback if regex fails - keep it as a raw string
                    anomaly_class = [anomaly_class]

            self.solution_dict[chan_id] = {
                'spacecraft': row['spacecraft'],
                'sequences': sequences,
                'class': anomaly_class,
                'num_values': int(row['num_values'])
            }
        
        print(f"--- Evaluation dictionary ready: {len(self.solution_dict)} channels processed ---")
        return self.solution_dict

    def compare_methods_results(self, predictions_dict):
        """
        Compares results from various methods against the processed solution_dict.
        
        Args:
            predictions_dict: Dictionary { 'chan_id': [list_of_outlier_indices] }
        """
        evaluation_results = []

        for chan_id, predicted_indices in predictions_dict.items():
            if chan_id not in self.solution_dict:
                continue

            gt_info = self.solution_dict[chan_id]
            num_values = gt_info['num_values']
            gt_sequences = gt_info['sequences']

            # Create binary masks for point-to-point comparison
            y_true = np.zeros(num_values)
            for start, end in gt_sequences:
                y_true[start : end + 1] = 1

            y_pred = np.zeros(num_values)
            # Filter indices to stay within bounds
            valid_indices = [i for i in predicted_indices if i < num_values]
            y_pred[valid_indices] = 1

            # Metric calculations
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            evaluation_results.append({
                'Channel': chan_id,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1_Score': round(f1, 4),
                'True_Points': int(np.sum(y_true)),
                'Pred_Points': int(np.sum(y_pred)),
                'TP': int(tp),
                'FP': int(fp),
                })
                 
        return pd.DataFrame(evaluation_results)
    
    def plot_hits_vs_misses(self, report_df):
        """
        Vytvoří sloupcový graf srovnávající počet tref (TP) a falešných poplachů (FP).
        
        Args:
            report_df: DataFrame, který vrací metoda compare_methods_results.
                       Musí obsahovat sloupce 'Channel', 'TP' a 'FP'.
        """
        channels = report_df['Channel']
        hits = report_df['TP']
        misses = report_df['FP']

        x = np.arange(len(channels))  # Pozice na ose X
        width = 0.35  # Šířka sloupců

        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Vykreslení dvou sloupců pro každý kanál
        rects1 = ax.bar(x - width/2, hits, width, label='Hits (True Positives)', color='forestgreen')
        rects2 = ax.bar(x + width/2, misses, width, label='Misses (False Positives)', color='crimson')

        # Popisky a design
        ax.set_ylabel('Počet datových bodů')
        ax.set_xlabel('Kanál (Soubor)')
        ax.set_title('Srovnání úspěšných zásahů a falešných poplachů podle PCA')
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45)
        ax.legend()

        # Přidání čísel nad sloupce (volitelné)
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()