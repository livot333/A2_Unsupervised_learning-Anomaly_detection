import numpy as np
import pandas as pd
import matplotlib as mtp 
import matplotlib.pyplot as plt 
import os
import seaborn as sns
import networkx as nx
import random


class DatasetOperations:
    def __init__(self,train_path, test_path,results_path):
        self.train_file_path = train_path
        self.test_file_path = test_path
        self.results_file_path = results_path

        self.train_data_dict = {}
        self.test_data_dict = {}
        self.labels_df = None

    def load_data(self,treshold_4_normalization = 1.5):
        '''
        loads dataset into three dictionaries: 
        1.train_data_dict
        2.test_data_dict
        3. labels_df 
        if values are above 1.5 normalizes them
        '''
        # 1. Load Training Data
        try:
            if not os.path.exists(self.train_file_path):
                raise FileNotFoundError(f"Directory not found: {self.train_file_path}")
                
            for file_name in os.listdir(self.train_file_path):
                if file_name.endswith('.npy'):
                    channel_id = file_name.replace('.npy', '')
                    path = os.path.join(self.train_file_path, file_name)
                    data = np.load(path)

                    telemetry = data[:, 0]
                    if np.max(np.abs(telemetry)) > treshold_4_normalization: # Pokud je hodnota vyšší než 2 (rezerva pro -1, 1)
                        print(f"  --> Scaling unnormalized channel: {channel_id} (Max value: {np.max(telemetry)})")
                        
                        # Min-Max Scaling do rozsahu [-1, 1]
                        t_min = np.min(telemetry)
                        t_max = np.max(telemetry)
                        
                        # Ošetření dělení nulou pro konstantní soubory
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

        # 2. Load Test Data
        try:
            if not os.path.exists(self.test_file_path):
                raise FileNotFoundError(f"Directory not found: {self.test_file_path}")

            for file_name in os.listdir(self.test_file_path):
                if file_name.endswith('.npy'):
                    channel_id = file_name.replace('.npy', '')
                    path = os.path.join(self.test_file_path, file_name)
                    data = np.load(path)

                    telemetry = data[:, 0]
                    if np.max(np.abs(telemetry)) > treshold_4_normalization: # Pokud je hodnota vyšší než 2 (rezerva pro -1, 1)
                        print(f"  --> Scaling unnormalized channel: {channel_id} (Max value: {np.max(telemetry)})")
                        
                        # Min-Max Scaling do rozsahu [-1, 1]
                        t_min = np.min(telemetry)
                        t_max = np.max(telemetry)
                        
                        # Ošetření dělení nulou pro konstantní soubory
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

        # 3. Load Labels CSV
        try:
            if os.path.exists(self.results_file_path):
                self.labels_df = pd.read_csv(self.results_file_path)
                if self.labels_df.empty:
                    print("Warning: Labels CSV is empty.")
                else:
                    print("Successfully loaded labels CSV.")
            else:
                print(f"Warning: Labels file not found at {self.results_file_path}")
        except Exception as e:
            print(f"Error loading results CSV: {e}")
    
        return self.train_data_dict,self.test_data_dict,self.labels_df
        

        
    def dataset_info(self):
        """Analyzes train and test dictionaries and displays summary tables."""
        
        def analyze_dict(data_dict, name):
            summary_list = []
            
            for channel_id, data in data_dict.items():
                rows, cols = data.shape
                
                # Check for missing values
                missing_values = np.isnan(data).sum()
                
                # Check binary values for Command Columns (Index 1 to End)
                command_cols = data[:, 1:]
                non_binary_count = np.sum((command_cols != 0) & (command_cols != 1))
                
                # Calculate range for Telemetry Column (Index 0)
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
                # Using to_string for better visibility of all columns
                print(info_df.to_string(index=False))
            else:
                print("No data loaded.")
            return info_df

        # Run analysis for both dictionaries
        analyze_dict(self.train_data_dict, "Training")
        analyze_dict(self.test_data_dict, "Testing")

        # Display Labels CSV preview
        print("\n--- LABELS CSV PREVIEW (First 5 rows) ---")
        if self.labels_df is not None:
            print(self.labels_df.head())
        else:
            print("Labels CSV not loaded.")



    def plot_data(self, choosen_dataset="training_dataset", start_channel_id="A-3"):
        """
        Interactive plot that allows browsing through channels using 'n' and 'b' keys.
        """
        # 1. Select the correct dictionary
        if choosen_dataset == "training_dataset":
            current_dict = self.train_data_dict
        else:
            current_dict = self.test_data_dict

        if not current_dict:
            print(f"Error: {choosen_dataset} dictionary is empty. Load data first.")
            return

        # 2. Prepare the list of keys and current index
        channel_keys = sorted(list(current_dict.keys()))
        try:
            self.current_idx = channel_keys.index(start_channel_id)
        except ValueError:
            self.current_idx = 0
            print(f"Warning: {start_channel_id} not found. Starting from index 0.")

        # 3. Setup the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.15)

        def update_plot():
            ax.clear()
            channel_id = channel_keys[self.current_idx]
            data = current_dict[channel_id]
            
            # Plot only the first column (Telemetry / Timeline)
            ax.plot(data[:, 0], color='blue', linewidth=1, label='Telemetry')
            
            ax.set_title(f"Mode: {choosen_dataset.upper()} | Channel: {channel_id} ({self.current_idx + 1}/{len(channel_keys)})")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Value")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            print(f"Displaying: {channel_id}")
            fig.canvas.draw()

        # 4. Key event handler
        def on_press(event):
            if event.key == 'n':
                self.current_idx = (self.current_idx + 1) % len(channel_keys)
                update_plot()
            elif event.key == 'b':
                self.current_idx = (self.current_idx - 1) % len(channel_keys)
                update_plot()
            elif event.key == 'escape':
                plt.close(fig)

        # 5. Connect event and show instructions
        fig.canvas.mpl_connect('key_press_event', on_press)
        print("\n--- Interactive Plotter Started ---")
        print("Controls: [n] Next channel | [b] Previous channel | [ESC] Close")
        
        update_plot()
        plt.show()



    def correlation_check(self, mode,correlation_csv_report = False, correlation_outfile_path = None, corr_calc_method="spearman"):
        """
        Performs correlation analysis and exports a structured 7-column CSV report.
        Format: Channel_ID | Internal_Type | Internal_Name | Internal_Value | External_Type | External_Name | External_Value
        """
        data_dict = self.train_data_dict if mode == "train" else self.test_data_dict
        if not data_dict:
            print("No data loaded to analyze.")
            return

        # 1. Inter-Channel Heatmap Visualization
        min_len = min(d.shape[0] for d in data_dict.values())
        inter_df = pd.DataFrame({cid: data_dict[cid][:min_len, 0] for cid in data_dict.keys()})
        inter_corr_matrix = inter_df.corr(method=corr_calc_method)

        plt.figure(figsize=(12, 10))
        sns.heatmap(inter_corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title(f"Inter-Channel Telemetry Correlation ({mode.upper()})")
        plt.show()
        self.inter_corr_matrix = inter_corr_matrix


        if correlation_csv_report == True:
            # 2. Building the Side-by-Side CSV report
            all_rows = []

            for channel_id, data in data_dict.items():
                # A. Internal: Telemetry vs Commands
                num_cols = data.shape[1]
                df_intra = pd.DataFrame(data, columns=['Telemetry'] + [f'Cmd_{i}' for i in range(1, num_cols)])
                intra_corr = df_intra.corr(method=corr_calc_method)['Telemetry'].drop('Telemetry')
                top_10_intra = intra_corr.abs().sort_values(ascending=False).head(5)

                df_intra = df_intra.loc[:, (df_intra != df_intra.iloc[0]).any()]


                # B. External: This Channel vs All Others
                inter_corr = inter_corr_matrix[channel_id].drop(channel_id)
                top_10_inter = inter_corr.abs().sort_values(ascending=False).head(5)

                # C. Merge both Top 10 lists into rows
                intra_list = list(top_10_intra.items())
                inter_list = list(top_10_inter.items())

                for i in range(5):
                    row = {'Channel_ID': channel_id if i == 0 else ''}
                    
                    # Internal Feature Columns
                    if i < len(intra_list):
                        feat_name, _ = intra_list[i]
                        row['Intra_Type'] = 'INTERNAL'
                        row['Intra_Feature'] = feat_name
                        row['Intra_Corr'] = intra_corr[feat_name]
                    else:
                        row['Intra_Type'], row['Intra_Feature'], row['Intra_Corr'] = '', '', ''

                    # External Channel Columns
                    if i < len(inter_list):
                        other_ch, _ = inter_list[i]
                        row['Inter_Type'] = 'EXTERNAL'
                        row['Inter_Channel'] = other_ch
                        row['Inter_Corr'] = inter_corr[other_ch]
                    else:
                        row['Inter_Type'], row['Inter_Channel'], row['Inter_Corr'] = '', '', ''

                    all_rows.append(row)

                # D. Add an empty row between channels for better visibility in Excel
                all_rows.append({k: '' for k in row.keys()})

            # 3. Exporting to CSV
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


    def sort_by_corr(self, sorting_threshold, remove_files = False, remove_threshold=None):
        """
        Uses an existing correlation matrix to group channels into clusters.
        """
        if self.inter_corr_matrix is None:
            print("Error: No correlation matrix provided for clustering.")
            return []

       
        G = nx.Graph()
        channels = self.inter_corr_matrix.columns
        G.add_nodes_from(channels)

        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                if abs(self.inter_corr_matrix.iloc[i, j]) >= sorting_threshold:
                    G.add_edge(channels[i], channels[j])

        # Získáme surové shluky
        raw_clusters = [list(c) for c in nx.connected_components(G)]
        raw_clusters.sort(key=len, reverse=True)

        final_clusters = []
        removed_count = 0

        # 2. Filtrace shluků (pokud je remove_files zapnuto)
        print(f"\n{'='*60}")
        print(f"CLUSTERING & CLEANING REPORT (Sort: {sorting_threshold} | Remove: {remove_threshold})")
        print(f"{'='*60}")

        for cluster in raw_clusters:
            if not remove_files:
                # Pokud nečistíme, necháme shluk tak, jak je
                final_clusters.append(cluster)
                continue
            
            # Čistící logika:
            leader = cluster[0]
            new_cluster = [leader] # Vůdce vždy zůstává
            
            for member in cluster[1:]:
                # Kontrolujeme korelaci člena vůči vůdci skupiny
                correlation_with_leader = abs(self.inter_corr_matrix.loc[leader, member])
                
                if correlation_with_leader >= remove_threshold:
                    # Soubor je redundantní -> smažeme ho z datasetů
                    if member in self.train_data_dict: del self.train_data_dict[member]
                    if member in self.test_data_dict: del self.test_data_dict[member]
                    removed_count += 1
                else:
                    # Soubor není dostatečně podobný vůdci -> necháme ho ve shluku
                    new_cluster.append(member)
            
            final_clusters.append(new_cluster)

        # 3. Výpis výsledků
        if remove_files:
            print(f"ACTION: Removed {removed_count} redundant files from memory.")
            print(f"RESULT: {len(self.train_data_dict)} channels remaining in dictionaries.")
        
        # Výpis složení (jen pro shluky, kde něco zbylo)
        for idx, c in enumerate([cl for cl in final_clusters if len(cl) > 1], 1):
            print(f" Group {idx:02d}: {', '.join(c)}")

        print(f"{'='*60}\n")
        
        return final_clusters
    

    def select_subset(self, random_selection=True, manual_file_names=None, subset_size=10, seed=42):
        """
        Vybere podmnožinu dat a vrátí dva slovníky: (train_subset, test_subset).
        """
        # Získáme všechny dostupné ID kanálů z trénovacích dat
        all_channels = list(self.train_data_dict.keys())
        
        if random_selection:
            # Fixní náhoda díky seedu
            random.seed(seed)
            actual_size = min(subset_size, len(all_channels))
            selected_keys = random.sample(all_channels, actual_size)
            print(f"Randomly selected {len(selected_keys)} files (seed={seed}).")
        
        elif manual_file_names:
            # Ruční výběr podle seznamu
            selected_keys = [f for f in manual_file_names if f in all_channels]
            missing = set(manual_file_names) - set(selected_keys)
            if missing:
                print(f"Warning: Files not found in dataset: {missing}")
            print(f"Manually selected {len(selected_keys)} files.")
        
        else:
            # Fallback: vezme prostě prvních N
            selected_keys = all_channels[:subset_size]
            print(f"Fallback: selected first {len(selected_keys)} files.")

        # Vytvoření nových pročištěných slovníků
        train_subset = {k: self.train_data_dict[k] for k in selected_keys if k in self.train_data_dict}
        test_subset = {k: self.test_data_dict[k] for k in selected_keys if k in self.test_data_dict}

        # Aktualizace vnitřního stavu třídy (aby i ostatní metody pracovaly s tímto výběrem)
        self.train_data_dict = train_subset
        self.test_data_dict = test_subset
        print(f"Selected files: {list(train_subset.keys())}")

        # VRACÍME DVA OBJEKTY
        return train_subset, test_subset