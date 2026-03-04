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

        total_tp = 0
        total_fp = 0
        total_fn = 0

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

            total_tp += tp
            total_fp += fp
            total_fn += fn

            evaluation_results.append({
                'Channel': chan_id,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1_Score': round(f1, 4),
                'True_Points': int(np.sum(y_true)),
                'Pred_Points': int(np.sum(y_pred)),
                'TP': int(tp),
                'FP': int(fp),
                'FN': int(fn)
                })
            

            report_df = pd.DataFrame(evaluation_results)
            macro_f1 = report_df['F1_Score'].mean()

            # 2. Micro F1 (Z globálních součtů)
            micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0

            print(f"\n=== Overall model evaluation ===")
            print(f"Macro-Average F1: {macro_f1:.4f}  (average sucessfull rate per chanel)")
            print(f"Micro-Average F1: {micro_f1:.4f}  (total sucess rate for all points)")
                 
        return report_df
    
    
    def plot_hits_vs_misses(self, report_df):
        """
        Creates a grouped bar chart comparing TP (Hits), FP (Misses), and FN (Missed).
        """
        # Data preparation
        channels = report_df['Channel']
        hits = report_df['TP']
        misses = report_df['FP']
        # We ensure FN is in the report_df or calculate it from other columns
        not_found = report_df['FN'] if 'FN' in report_df.columns else (report_df['True_Points'] - report_df['TP'])

        x = np.arange(len(channels))  # Label locations
        width = 0.25  # Width of each bar - reduced to fit 3 side-by-side

        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plotting the three bars with high-quality colors
        # TP = Success (Green), FP = False Alarm (Red), FN = Missed Reality (Blue)
        rects1 = ax.bar(x - width, hits, width, label='Hits (True Positives)', color='#2ca02c', edgecolor='white')
        rects2 = ax.bar(x, misses, width, label='Misses (False Positives)', color='#d62728', edgecolor='white')
        rects3 = ax.bar(x + width, not_found, width, label='Missed (False Negatives)', color='#1f77b4', edgecolor='white')

        # Text and Styling (All in English)
        ax.set_ylabel('Data Point Count', fontsize=12, fontweight='bold')
        ax.set_xlabel('Channel ID', fontsize=12, fontweight='bold')
        ax.set_title('Anomaly Detection Performance: PCA Reconstruction Method', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha='right')
        ax.legend(fontsize=11, frameon=True, shadow=True)

        # Adding values on top of bars for precision
        ax.bar_label(rects1, padding=3, fontsize=9, color='#1a1a1a')
        ax.bar_label(rects2, padding=3, fontsize=9, color='#1a1a1a')
        ax.bar_label(rects3, padding=3, fontsize=9, color='#1a1a1a')

        # Aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        plt.show()