import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import re

class EvaluateResults:
    def __init__(self):
        """
        Initializes the evaluation object with an empty solution dictionary.
        """
        self.solution_dict = {}

    def load_solution(self, labels_df):
        """
        Processes the labels DataFrame and creates a master dictionary of ground truth anomaly intervals.
        """
        if labels_df is None or labels_df.empty:
            print("Error: Labels DataFrame is empty.")
            return None

        # Reset dictionary to ensure fresh load
        self.solution_dict = {}

        # Ensure column names are stripped of whitespace
        labels_df.columns = labels_df.columns.str.strip()

        for _, row in labels_df.iterrows():
            # Skip rows with missing essential information
            if pd.isna(row['chan_id']) or pd.isna(row['num_values']):
                continue

            chan_id = str(row['chan_id']).strip()
            
            try:
                # 1. Parse anomaly sequences (start and end index intervals)
                sequences = row['anomaly_sequences']
                if isinstance(sequences, str):
                    sequences = ast.literal_eval(sequences)
                
                # 2. Parse and sanitize the 'class' column strings
                anomaly_class = row['class']
                if isinstance(anomaly_class, str):
                    try:
                        # Wrap unquoted words in quotes for literal_eval parsing
                        sanitized_class = re.sub(r'([a-zA-Z_]\w*)', r'"\1"', anomaly_class)
                        anomaly_class = ast.literal_eval(sanitized_class)
                    except Exception:
                        anomaly_class = [anomaly_class]

                # 3. Store metadata and ground truth in the master dictionary
                self.solution_dict[chan_id] = {
                    'spacecraft': row['spacecraft'],
                    'sequences': sequences,
                    'class': anomaly_class,
                    'num_values': int(float(row['num_values'])) # float conversion handles string decimals like '2264.0'
                }
            except Exception:
                # Silently skip rows that encounter parsing errors
                continue
        
        print(f"--- Evaluation dictionary ready: {len(self.solution_dict)} channels processed ---")
        return self.solution_dict

    def compare_methods_results(self, predictions_dict, total_lengths_dict=None):
        """
        Universal evaluation method: works with both raw data and sliding window outputs.
        """
        evaluation_results = []
        total_tp, total_fp, total_fn = 0, 0, 0

        for chan_id, predicted_indices in predictions_dict.items():
            # Match the channel ID (handling potential file extensions)
            clean_id = str(chan_id).replace('.csv', '').strip()
            
            if clean_id not in self.solution_dict:
                # Skip if the channel is not present in the ground truth labels
                continue

            gt_info = self.solution_dict[clean_id]
            num_values = gt_info['num_values']
            
            # Use shortened length for sliding window alignment, otherwise use full length
            m = total_lengths_dict[clean_id] if (total_lengths_dict and clean_id in total_lengths_dict) else num_values
            drop_count = num_values - m

            # Create binary Ground Truth masks
            y_true_full = np.zeros(num_values)
            for start, end in gt_info['sequences']:
                y_true_full[start : end + 1] = 1
            y_true = y_true_full[drop_count:] # Alignment with processed data length

            # Create binary Prediction masks
            y_pred = np.zeros(m)
            valid_indices = [i for i in predicted_indices if i < m]
            y_pred[valid_indices] = 1

            # --- POINT ADJUST LOGIC ---
            # Standard in time-series: if any point in an anomaly segment is detected, 
            # the entire segment is considered detected.
            
            y_pred_adjusted = y_pred.copy()
            for start, end in gt_info['sequences']:
                adj_start = max(0, start - drop_count)
                adj_end = max(0, end - drop_count)
                if np.any(y_pred[adj_start : adj_end + 1] == 1):
                    y_pred_adjusted[adj_start : adj_end + 1] = 1

            # Calculate basic Confusion Matrix metrics
            
            tp = np.sum((y_true == 1) & (y_pred_adjusted == 1))
            fp = np.sum((y_true == 0) & (y_pred_adjusted == 1))
            fn = np.sum((y_true == 1) & (y_pred_adjusted == 0))

            # Calculate scores
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Update global counters for Micro-averaging
            total_tp += tp
            total_fp += fp
            total_fn += fn

            evaluation_results.append({
                'Channel': clean_id,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1_Score': round(f1, 4),
                'TP': int(tp), 'FP': int(fp), 'FN': int(fn),
                'True_Points': int(np.sum(y_true)),
                'Pred_Points': int(np.sum(y_pred_adjusted))
            })

        if not evaluation_results:
            print("Warning: No matching channels found between predictions and solution!")
            return pd.DataFrame()

        report_df = pd.DataFrame(evaluation_results)
        
        # Calculate Macro and Micro F1 scores
        macro_f1 = report_df['F1_Score'].mean()
        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0

        print(f"\n=== Evaluation: Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f} ===")
        return report_df

    def plot_hits_vs_misses(self, report_df, title=None):
        """
        Visualizes performance using a grouped bar chart comparing TP (Hits), FP (Misses), and FN (False Negatives).
        """
        # Prepare plotting data
        channels = report_df['Channel']
        hits = report_df['TP']
        misses = report_df['FP']
        
        # Calculate False Negatives if not explicitly provided
        not_found = report_df['FN'] if 'FN' in report_df.columns else (report_df['True_Points'] - report_df['TP'])

        x = np.arange(len(channels))  # Bar position labels
        width = 0.25  # Individual bar width

        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create bars for TP, FP, and FN
        rects1 = ax.bar(x - width, hits, width, label='Hits (True Positives)', color='#2ca02c', edgecolor='white')
        rects2 = ax.bar(x, misses, width, label='Misses (False Positives)', color='#d62728', edgecolor='white')
        rects3 = ax.bar(x + width, not_found, width, label='Missed (False Negatives)', color='#1f77b4', edgecolor='white')

        # Chart Labels and Styling
        ax.set_ylabel('Data Point Count', fontsize=12, fontweight='bold')
        ax.set_xlabel('Channel ID', fontsize=12, fontweight='bold')
        ax.set_title(title or 'Anomaly Detection Performance', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha='right')
        ax.legend(fontsize=11, frameon=True, shadow=True)

        # Add data labels on top of bars
        ax.bar_label(rects1, padding=3, fontsize=9, color='#1a1a1a')
        ax.bar_label(rects2, padding=3, fontsize=9, color='#1a1a1a')
        ax.bar_label(rects3, padding=3, fontsize=9, color='#1a1a1a')

        # Plot aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        
        fig.tight_layout()
        plt.show()