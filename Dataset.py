import numpy as np
import pandas as pd
import matplotlib as mtp 
import matplotlib.pyplot as plt 
import os
import seaborn as sns
import networkx as nx
import random
import io

class DatasetOperations:
    def __init__(self, train_path, test_path, results_path):
        """
        Initializes the dataset operation helper.
        :param train_path: System path to training .npy files
        :param test_path: System path to testing .npy files
        :param results_path: Path to the labeled anomalies CSV file
        """
        self.train_file_path = train_path
        self.test_file_path = test_path
        self.results_file_path = results_path

        self.train_data_dict = {}
        self.test_data_dict = {}
        self.labels_df = None

    def load_data(self, treshold_4_normalization=1.5):
        """
        Loads dataset into three storage structures: 
        1. train_data_dict
        2. test_data_dict
        3. labels_df 
        Applies Min-Max normalization if telemetry values exceed the specified threshold.
        """
        # --- 1. Load Training Data ---
        try:
            if not os.path.exists(self.train_file_path):
                raise FileNotFoundError(f"Directory not found: {self.train_file_path}")
                
            for file_name in os.listdir(self.train_file_path):
                if file_name.endswith('.npy'):
                    channel_id = file_name.replace('.npy', '')
                    path = os.path.join(self.train_file_path, file_name)
                    data = np.load(path)

                    telemetry = data[:, 0]
                    # Check if telemetry exceeds threshold for scaling
                    if np.max(np.abs(telemetry)) > treshold_4_normalization: 
                        print(f"  --> Scaling unnormalized channel: {channel_id} (Max value: {np.max(telemetry)})")
                        
                        # Apply Min-Max Scaling to range [-1, 1]
                        t_min = np.min(telemetry)
                        t_max = np.max(telemetry)
                        
                        # Avoid division by zero for constant signals
                        if t_max - t_min != 0:
                            data[:, 0] = ((telemetry - t_min) / (t_max - t_min)) * 2 - 1
                        else:
                            data[:, 0] = 0
                    
                    if data.size == 0:
                        print(f"Warning: {file_name} is empty.")
                    else:
                        self.train_data_dict[channel_id] = data

            print(f"Successfully loaded {len(self.train_data_dict)} train channels.")
        except Exception as e:
            print(f"Error loading train data: {e}")

        # --- 2. Load Test Data ---
        try:
            if not os.path.exists(self.test_file_path):
                raise FileNotFoundError(f"Directory not found: {self.test_file_path}")

            for file_name in os.listdir(self.test_file_path):
                if file_name.endswith('.npy'):
                    channel_id = file_name.replace('.npy', '')
                    path = os.path.join(self.test_file_path, file_name)
                    data = np.load(path)

                    telemetry = data[:, 0]
                    if np.max(np.abs(telemetry)) > treshold_4_normalization:
                        print(f"  --> Scaling unnormalized channel: {channel_id} (Max value: {np.max(telemetry)})")
                        
                        t_min = np.min(telemetry)
                        t_max = np.max(telemetry)
                        
                        if t_max - t_min != 0:
                            data[:, 0] = ((telemetry - t_min) / (t_max - t_min)) * 2 - 1
                        else:
                            data[:, 0] = 0
                                    
                    if data.size == 0:
                        print(f"Warning: {file_name} is empty.")
                    else:
                        self.test_data_dict[channel_id] = data
            print(f"Successfully loaded {len(self.test_data_dict)} test channels.")
        except Exception as e:
            print(f"Error loading test data: {e}")

        # --- 3. Load Labels CSV ---
        try:
            with open(self.results_file_path, 'r') as f:
                lines = f.readlines()

            header = lines[0].strip()
            fixed_lines = [header]

            for line in lines[1:]:
                line = line.strip()
                # Clean nested quotes from the NASA labels format
                if line.startswith('"') and line.endswith('"'):
                    line = line[1:-1]
                # Convert escaped double quotes to single quotes
                line = line.replace('""', '"')
                fixed_lines.append(line)

            self.labels_df = pd.read_csv(io.StringIO('\n'.join(fixed_lines)))
        except Exception as e:
            print(f"Error loading results CSV: {e}")
    
        return self.train_data_dict, self.test_data_dict, self.labels_df
        
    def dataset_info(self):
        """Analyzes train and test dictionaries and displays summary tables."""
        
        def analyze_dict(data_dict, name):
            summary_list = []
            
            for channel_id, data in data_dict.items():
                rows, cols = data.shape
                
                # Identify missing data points
                missing_values = np.isnan(data).sum()
                
                # Check for validity of binary Command Columns (Index 1 onwards)
                command_cols = data[:, 1:]
                non_binary_count = np.sum((command_cols != 0) & (command_cols != 1))
                
                # Observe value distribution for Telemetry Column (Index 0)
                telemetry_col = data[:, 0]
                min_val = np.min(telemetry_col)
                max_val = np.max(telemetry_col)
                
                summary_list.append({
                    'Channel_ID': channel_id,
                    'Dimensions': f"{rows} x {cols}",
                    'Telemetry_Range': f"[{min_val:.2f}, {max_val:.2f}]",
                    'Missing_NaN': missing_values,
                    'Non_Binary_Cmds': non_binary_count
                })
            
            info_df = pd.DataFrame(summary_list)
            print(f"\n--- {name.upper()} DATASET SUMMARY ---")
            if not info_df.empty:
                print(info_df.to_string(index=False))
            else:
                print("No data loaded.")
            return info_df

        # Perform analysis for both data subsets
        analyze_dict(self.train_data_dict, "Training")
        analyze_dict(self.test_data_dict, "Testing")

        # Preview of the labeled anomalies
        print("\n--- LABELS CSV PREVIEW (First 5 rows) ---")
        if self.labels_df is not None:
            print(self.labels_df.head())
        else:
            print("Labels CSV not loaded.")

    def plot_data(self, choosen_dataset="testing_dataset", start_channel_id="P-1"):
        """
        Interactive plot that allows browsing through channels using 'n' (Next) and 'b' (Back) keys.
        """
        # Select target dataset
        if choosen_dataset == "training_dataset":
            current_dict = self.train_data_dict
        else:
            current_dict = self.test_data_dict

        if not current_dict:
            print(f"Error: {choosen_dataset} dictionary is empty. Load data first.")
            return

        # Map keys for navigation
        channel_keys = sorted(list(current_dict.keys()))
        try:
            self.current_idx = channel_keys.index(start_channel_id)
        except ValueError:
            self.current_idx = 0
            print(f"Warning: {start_channel_id} not found. Starting from index 0.")

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.15)

        def update_plot():
            ax.clear()
            channel_id = channel_keys[self.current_idx]
            data = current_dict[channel_id]
            
            # Focus on the first column representing Telemetry
            ax.plot(data[:, 0], color='blue', linewidth=1, label='Telemetry')
            
            ax.set_title(f"Mode: {choosen_dataset.upper()} | Channel: {channel_id} ({self.current_idx + 1}/{len(channel_keys)})")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Value")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            print(f"Displaying: {channel_id}")
            fig.canvas.draw()

        def on_press(event):
            # Handler for keyboard shortcuts
            if event.key == 'n':
                self.current_idx = (self.current_idx + 1) % len(channel_keys)
                update_plot()
            elif event.key == 'b':
                self.current_idx = (self.current_idx - 1) % len(channel_keys)
                update_plot()
            elif event.key == 'escape':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_press)
        print("\n--- Interactive Plotter Started ---")
        print("Controls: [n] Next channel | [b] Previous channel | [ESC] Close")
        
        update_plot()
        plt.show()


    def plot_data_signal(self, choosen_dataset="testing_dataset", channel_id="P-1"):
        """
        Plot the telemetry signal for a single channel.
        """
        if choosen_dataset == "training_dataset":
            current_dict = self.train_data_dict
        else:
            current_dict = self.test_data_dict

        if not current_dict:
            print(f"Error: {choosen_dataset} dictionary is empty. Load data first.")
            return

        if channel_id not in current_dict:
            print(f"Warning: {channel_id} not found.")
            return

        data = current_dict[channel_id]

        _, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data[:, 0], color='blue', linewidth=1, label='Telemetry')
        ax.set_title(f"Mode: {choosen_dataset.upper()} | Channel: {channel_id}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Normalized Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()

    def correlation_check(self, mode, correlation_csv_report=False, correlation_outfile_path=None, corr_calc_method="spearman"):
        """
        Performs correlation analysis and exports a structured 7-column CSV report.
        Includes Inter-Channel heatmaps and specific Internal Feature correlations.
        """
        data_dict = self.train_data_dict if mode == "train" else self.test_data_dict
        if not data_dict:
            print("No data loaded to analyze.")
            return

        # --- 1. Inter-Channel Heatmap Visualization ---
        # Align lengths by trimming to the shortest sequence
        min_len = min(d.shape[0] for d in data_dict.values())
        inter_df = pd.DataFrame({cid: data_dict[cid][:min_len, 0] for cid in data_dict.keys()})
        inter_corr_matrix = inter_df.corr(method=corr_calc_method)

        plt.figure(figsize=(12, 10))
        sns.heatmap(inter_corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title(f"Inter-Channel Telemetry Correlation ({mode.upper()})")
        plt.show()
        self.inter_corr_matrix = inter_corr_matrix

        if correlation_csv_report:
            # --- 2. Build Side-by-Side CSV report ---
            all_rows = []

            for channel_id, data in data_dict.items():
                # A. Internal correlation: Telemetry vs Binary Command signals
                num_cols = data.shape[1]
                df_intra = pd.DataFrame(data, columns=['Telemetry'] + [f'Cmd_{i}' for i in range(1, num_cols)])
                
                # Exclude constant columns to prevent NaNs in correlation
                df_intra = df_intra.loc[:, (df_intra != df_intra.iloc[0]).any()]
                intra_corr = df_intra.corr(method=corr_calc_method)['Telemetry'].drop('Telemetry', errors='ignore')
                top_5_intra = intra_corr.abs().sort_values(ascending=False).head(5)

                # B. External correlation: Comparing this channel against all other telemetry channels
                inter_corr = inter_corr_matrix[channel_id].drop(channel_id)
                top_5_inter = inter_corr.abs().sort_values(ascending=False).head(5)

                # C. Mapping Top values to the report rows
                intra_list = list(top_5_intra.items())
                inter_list = list(top_5_inter.items())

                for i in range(5):
                    row = {'Channel_ID': channel_id if i == 0 else ''}
                    
                    # Log Internal Feature Correlations
                    if i < len(intra_list):
                        feat_name, _ = intra_list[i]
                        row['Intra_Type'] = 'INTERNAL'
                        row['Intra_Feature'] = feat_name
                        row['Intra_Corr'] = intra_corr[feat_name]
                    else:
                        row['Intra_Type'], row['Intra_Feature'], row['Intra_Corr'] = '', '', ''

                    # Log External Channel Correlations
                    if i < len(inter_list):
                        other_ch, _ = inter_list[i]
                        row['Inter_Type'] = 'EXTERNAL'
                        row['Inter_Channel'] = other_ch
                        row['Inter_Corr'] = inter_corr[other_ch]
                    else:
                        row['Inter_Type'], row['Inter_Channel'], row['Inter_Corr'] = '', '', ''

                    all_rows.append(row)

                # Add separator row for CSV readability
                all_rows.append({k: '' for k in row.keys()})

            # --- 3. Export to File ---
            report_df = pd.DataFrame(all_rows)
            full_path = os.path.join(correlation_outfile_path, 'correlation_report.csv')
                    
            report_df.to_csv(
                full_path, 
                index=False, 
                sep=";", 
                decimal=",", 
                escapechar=";"
            )
            print(f"Detailed correlation report saved to: {full_path}")
            
            return report_df

    def sort_by_corr(self, sorting_threshold, remove_files=False, remove_threshold=None):
        """
        Clusters channels into groups based on their correlation matrix.
        Can optionally remove highly redundant files to optimize memory.
        """
        if self.inter_corr_matrix is None:
            print("Error: No correlation matrix provided for clustering.")
            return []

        # Construct adjacency graph where edges exist if correlation exceeds threshold
        G = nx.Graph()
        channels = self.inter_corr_matrix.columns
        G.add_nodes_from(channels)

        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                if abs(self.inter_corr_matrix.iloc[i, j]) >= sorting_threshold:
                    G.add_edge(channels[i], channels[j])

        # Extract raw clusters as connected components
        raw_clusters = [list(c) for c in nx.connected_components(G)]
        raw_clusters.sort(key=len, reverse=True)

        final_clusters = []
        removed_count = 0

        print(f"\n{'='*60}")
        print(f"CLUSTERING & CLEANING REPORT (Sort: {sorting_threshold} | Remove: {remove_threshold})")
        print(f"{'='*60}")

        for cluster in raw_clusters:
            if not remove_files:
                final_clusters.append(cluster)
                continue
            
            # Cleaning Logic: Keep the cluster "Leader" and remove redundant members
            leader = cluster[0]
            new_cluster = [leader] 
            
            for member in cluster[1:]:
                correlation_with_leader = abs(self.inter_corr_matrix.loc[leader, member])
                
                if correlation_with_leader >= remove_threshold:
                    # File is redundant -> purge from memory
                    if member in self.train_data_dict: del self.train_data_dict[member]
                    if member in self.test_data_dict: del self.test_data_dict[member]
                    removed_count += 1
                else:
                    # Member differs enough from leader -> keep in group
                    new_cluster.append(member)
            
            final_clusters.append(new_cluster)

        if remove_files:
            print(f"ACTION: Removed {removed_count} redundant files from memory.")
            print(f"RESULT: {len(self.train_data_dict)} channels remaining in dictionaries.")
        
        # Log group composition
        for idx, c in enumerate([cl for cl in final_clusters if len(cl) > 1], 1):
            print(f" Group {idx:02d}: {', '.join(c)}")

        print(f"{'='*60}\n")
        
        return final_clusters

    def select_subset(self, random_selection=True, manual_file_names=None, subset_size=10, seed=42):
        """
        Slices the dataset to a specific subset.
        Returns: (train_subset, test_subset)
        """
        all_channels = list(self.train_data_dict.keys())
        
        if random_selection:
            # Deterministic randomness using a seed
            random.seed(seed)
            actual_size = min(subset_size, len(all_channels))
            selected_keys = random.sample(all_channels, actual_size)
            print(f"Randomly selected {len(selected_keys)} files (seed={seed}).")
        
        elif manual_file_names:
            # User defined list of IDs
            selected_keys = [f for f in manual_file_names if f in all_channels]
            missing = set(manual_file_names) - set(selected_keys)
            if missing:
                print(f"Warning: Files not found in dataset: {missing}")
            print(f"Manually selected {len(selected_keys)} files.")
        
        else:
            # Selection of the first N files
            selected_keys = all_channels[:subset_size]
            print(f"Fallback: selected first {len(selected_keys)} files.")

        # Reconstruct internal dictionaries based on selection
        train_subset = {k: self.train_data_dict[k] for k in selected_keys if k in self.train_data_dict}
        test_subset = {k: self.test_data_dict[k] for k in selected_keys if k in self.test_data_dict}

        # Sync the internal state of the class
        self.train_data_dict = train_subset
        self.test_data_dict = test_subset
        print(f"Selected files: {list(train_subset.keys())}")

        return train_subset, test_subset

    def remove_constant_columns(self, train_dict, test_dict):
        """
        Removes columns that are constant in both Training AND Testing datasets.
        Ensures both resulting datasets have an identical feature structure (dimensionality).
        """
        print("--- Synchronized removal of constant columns ---")
        new_train_dict = {}
        new_test_dict = {}
        
        # Intersect IDs to ensure alignment
        common_cids = set(train_dict.keys()).intersection(set(test_dict.keys()))
        
        for cid in common_cids:
            train_df = pd.DataFrame(train_dict[cid])
            test_df = pd.DataFrame(test_dict[cid])
            
            # Identify columns with variance in Training
            train_moving = train_df.nunique() > 1
            
            # Identify columns with variance in Testing
            test_moving = test_df.nunique() > 1
            
            # MERGED MASK: Keep column if it moves in either Train OR Test
            # (Purge only if it is dead/constant in both sets)
            keep_cols_mask = train_moving | test_moving
            
            cols_to_keep = train_df.columns[keep_cols_mask]
            
            new_train_dict[cid] = train_df[cols_to_keep].values
            new_test_dict[cid] = test_df[cols_to_keep].values
            
            removed_count = train_df.shape[1] - len(cols_to_keep)
            if removed_count > 0:
                print(f" Channel {cid:6}: Removed {removed_count} columns ")
                
        return new_train_dict, new_test_dict