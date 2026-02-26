from Dataset import DatasetOperations 

train_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\test"
test_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\train"
result_file_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\labeled_anomalies.csv"


dt = DatasetOperations(test_path=test_path,train_path=train_path,results_path=result_file_path)

train_dataset, test_dataset , results = dt.load_data()
dt.dataset_info()
dt_plt = dt.plot_data(choosen_dataset="training_dataset",start_channel_id="A1")
