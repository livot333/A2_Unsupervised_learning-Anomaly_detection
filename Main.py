from Dataset import DatasetOperations 

train_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\test"
test_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\train"
result_file_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\labeled_anomalies.csv"

correlation_outfile_path = r"D:\SKOLA\NTNU\MLL\Assingment_2"


dt = DatasetOperations(test_path=test_path,train_path=train_path,results_path=result_file_path)

train_dataset, test_dataset , results = dt.load_data()
dt.dataset_info()
dt_plt = dt.plot_data(choosen_dataset="training_dataset",start_channel_id="A1")
correlation_dictionary = dt.correlation_check(mode="training_dataset",correlation_csv_report=True,correlation_outfile_path=correlation_outfile_path,corr_calc_method="spearman")
train_data_clustered_by_corr = dt.sort_by_corr(sorting_threshold=0.60, remove_files=False,remove_threshold=None)