from Dataset import DatasetOperations 
from PCA_method import BatchPCA
import os
from Evaluation import EvaluateResults

train_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\test"
test_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\train"
result_file_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\labeled_anomalies.csv"

correlation_outfile_path = r"D:\SKOLA\NTNU\MLL\Assingment_2"


# ========================================== Data load and prepare =========================================
dt = DatasetOperations(test_path=test_path,train_path=train_path,results_path=result_file_path)

train_dataset, test_dataset , results = dt.load_data()

# dt.dataset_info()
# dt_plt = dt.plot_data(choosen_dataset="training_dataset",start_channel_id="P-1")
# correlation_dictionary = dt.correlation_check(mode="training_dataset",correlation_csv_report=False,correlation_outfile_path=correlation_outfile_path,corr_calc_method="spearman")
# train_data_clustered_by_corr = dt.sort_by_corr(sorting_threshold=0.60, remove_files=False,remove_threshold=None)
train_subset,test_subset = dt.select_subset(random_selection=True, manual_file_names=None, subset_size=10, seed=42)
# train_subset_cl = dt.remove_constant_columns(train_subset)
# test_subset_cl = dt.remove_constant_columns(test_subset)
evaluation = EvaluateResults()
evaluation.load_solution(results)

# ===================================== PCA ====================================
bpca = BatchPCA(train_subset, test_subset)
bpca.fit_all(n_components=20)
PCA_test_errors = bpca.get_PCA_predictions(mode="test",threshold_percentile=50)
PCA_result = evaluation.compare_methods_results(predictions_dict=PCA_test_errors)
evaluation.plot_hits_vs_misses(PCA_result)
print(PCA_result)
# ===============================================================================

# # 5. Podíváš se na výsledky
# bpca.plot_summary(test_errors)
# evaluation_report_pca = bpca.evaluate_pca_anomalies(test_errors, results, threshold_percentile=98)
# print(evaluation_report_pca)

