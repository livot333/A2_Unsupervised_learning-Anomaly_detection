from Dataset import DatasetOperations 

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_path =  join(BASE_DIR, "archive", "data", "data", "train")
test_path = os.path.join(BASE_DIR, "archive", "data", "data", "test")
result_file_path = os.path.join(BASE_DIR, "archive", "labeled_anomalies.csv")

correlation_outfile_path = BASE_DIR


dt = DatasetOperations(test_path=test_path,train_path=train_path,results_path=result_file_path)

train_dataset, test_dataset , results = dt.load_data()
dt.dataset_info()
dt_plt = dt.plot_data(choosen_dataset="training_dataset",start_channel_id="A1")
correlation_dictionary = dt.correlation_check(mode="training_dataset",correlation_csv_report=True,correlation_outfile_path=correlation_outfile_path,corr_calc_method="spearman")
train_data_clustered_by_corr = dt.sort_by_corr(sorting_threshold=0.60, remove_files=False,remove_threshold=None)