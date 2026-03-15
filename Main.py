from Dataset import DatasetOperations 
from PCA_method import BatchPCA
import os
from Evaluation import EvaluateResults
from GausianMixtureModel import GMM
from LocalOutlierFactor import LOF
from TimeContext import TimeContextModif
import numpy as np
from Stompy import MstumpDetector
from LSTMAutoencoder import LSTM_AE_Detector

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

#================================= TIme context =================================
tx = TimeContextModif(test_dataset=test_subset,train_dataset=train_subset)
train_s, test_s = tx.apply_sliding_window(window_length=100, flatten=True)
# train_s, test_s = tx.add_lag_features(lags=np.arange(1, 10, 1))
# train_s, test_s = tx.add_rolling_statistics(window_length=500)
# train_s, test_s = tx.add_spectral_features(window_length=500)
# train_s, test_s = tx.add_derivative_features()
# ===================================== PCA ====================================
# bpca = BatchPCA(train_subset, test_subset)
# bpca.fit_all(n_components=20)
# PCA_test_errors = bpca.get_PCA_predictions(mode="test",threshold_percentile=50)
# PCA_result = evaluation.compare_methods_results(predictions_dict=PCA_test_errors)
# evaluation.plot_hits_vs_misses(PCA_result)
# print(PCA_result)
# # ===================================== GMM ======================================

gmm = GMM(train_subset,test_subset)
gmm.fit_all(n_components=10,covariance_type='full',n_init=50, max_iter=4500, tol=1e-2)
GMM_errors_pred = gmm.get_batch_predictions(threshold_percentile=5)
GMM_results = evaluation.compare_methods_results(predictions_dict=GMM_errors_pred)
evaluation.plot_hits_vs_misses(GMM_results)
print(GMM_results)

# # ====================================== LOF ========================================

# lof = LOF(train_s,test_s)
# lof.fit_all(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, contamination='auto')
# LOF_error_prediction = lof.get_batch_predictions(threshold_percentile=10)
# LOF_results = evaluation.compare_methods_results(predictions_dict=LOF_error_prediction)
# evaluation.plot_hits_vs_misses(LOF_results)
# print(LOF_results)


#===================================STOMPY===========================================
# stmp = MstumpDetector(window_size=200)
# stmp_error_prediction = stmp.get_batch_predictions(test_subset, threshold_percentile=99.5)
# stmp_report = evaluation.compare_methods_results(stmp_error_prediction)
# evaluation.plot_hits_vs_misses(stmp_report)
# print(stmp_report)

# ================================ LSTMA ===========================================

lstm_ae = LSTM_AE_Detector(seq_len=250, epochs=10, percentile=94)

# Fit (na celém slovníku)
lstm_ae.fit(train_subset)

# Prediction (dostaneš slovník množin)
ae_outliers = lstm_ae.prediction(test_subset)

# Tvůj stávající report
report_lstma = evaluation.compare_methods_results(ae_outliers)
evaluation.plot_hits_vs_misses(report_lstma)
print(report_lstma)