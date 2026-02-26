import numpy as np
import pandas as pd
import matplotlib as mtp 
import matplotlib.pyplot as plt 
import os


class DatasetOperations:
    def __init__(self,train_path, test_path,results_path):
        self.train_file_path = train_path
        self.test_file_path = test_path
        self.results_file_path = results_path

        self.train_data_dict = {}
        self.test_data_dict = {}
        self.labels_df = None

    def load_data(self):
        '''
        loads dataset into three dictionaries: 
        1.train_data_dict
        2.test_data_dict
        3. labels_df 
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

        
