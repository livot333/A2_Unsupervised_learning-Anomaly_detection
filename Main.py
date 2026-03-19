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
from AutoEncoder import MultiFileSklearnTuner
import optuna


train_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\train"
test_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\data\data\test"
result_file_path = r"D:\SKOLA\NTNU\MLL\Assingment_2\Dataset_file\labeled_anomalies.csv"

correlation_outfile_path = r"D:\SKOLA\NTNU\MLL\Assingment_2"


# ========================================== Data load and prepare =========================================
dt = DatasetOperations(test_path=test_path,train_path=train_path,results_path=result_file_path)

train_dataset, test_dataset , results = dt.load_data()

# dt.dataset_info()
# dt_plt = dt.plot_data(choosen_dataset="training_dataset",start_channel_id="P-1")
# correlation_dictionary = dt.correlation_check(mode="training_dataset",correlation_csv_report=False,correlation_outfile_path=correlation_outfile_path,corr_calc_method="spearman")
# train_data_clustered_by_corr = dt.sort_by_corr(sorting_threshold=0.60, remove_files=True,remove_threshold=98)
train_subset,test_subset = dt.select_subset(random_selection=True, manual_file_names=None, subset_size=30, seed=19)
train_subset_cl,test_subset_cl = dt.remove_constant_columns(train_subset,test_dict=test_subset)
evaluation = EvaluateResults()
evaluation.load_solution(results)

#================================= TIme context =================================
tx = TimeContextModif(test_dataset=test_subset_cl,train_dataset=train_subset_cl)
train_s_der, test_s_der = tx.add_derivative_features()

tcx = TimeContextModif(test_dataset=test_s_der,train_dataset=train_s_der)
train_s, test_s = tcx.add_spectral_features(window_length=250)

# tcx2 = TimeContextModif(test_dataset=test_s_fft,train_dataset=train_s_fft)
# train_s_sw, test_s_sw = tcx2.apply_sliding_window(window_length=220, flatten=True)

# tcx3 = TimeContextModif(test_dataset=test_s_sw,train_dataset=train_s_sw)
# train_s_lw, test_s_lw = tcx3.add_lag_features(lags=np.arange(1, 5, 1))

# tcx4 = TimeContextModif(test_dataset=test_s_sw,train_dataset=train_s_sw)
# train_s, test_s = tcx4.add_rolling_statistics(window_length=200)

# train_s, test_s = tcx.add_rolling_statistics(window_length=100)

# ===================================== PCA ====================================
# sorting 
bpca_sort = BatchPCA(train_s, test_s)
bpca_sort.fit_all(n_components=30)
PCA_sort_test_errors = bpca_sort.get_PCA_predictions(mode="test",threshold_percentile=80)
PCA_sort_result = evaluation.compare_methods_results(predictions_dict=PCA_sort_test_errors)

surviving_channels = PCA_sort_result[PCA_sort_result['TP'] > 0]['Channel'].tolist()
train_surv = {cid: train_s[cid] for cid in surviving_channels}
test_surv = {cid: test_s[cid] for cid in surviving_channels}

#prep
bpca = BatchPCA(train_surv, test_surv)
bpca.fit_all(n_components=15)
PCA_test_errors = bpca.get_PCA_predictions(mode="test",threshold_percentile=95)
PCA_result = evaluation.compare_methods_results(predictions_dict=PCA_test_errors)
# evaluation.plot_hits_vs_misses(PCA_result)
# print(PCA_result)
train_pca_features = bpca.transform_PCA(mode="train")
test_pca_features = bpca.transform_PCA(mode="test")



# # ===================================== GMM ======================================

gmm = GMM(train_pca_features,test_pca_features)
gmm.fit_all(n_components=3, covariance_type='diag', n_init=5, max_iter=300, tol=1e-3, init_params='kmeans', warm_start=True)
GMM_errors_pred = gmm.get_batch_predictions(threshold_percentile=5)
GMM_results = evaluation.compare_methods_results(predictions_dict=GMM_errors_pred)
evaluation.plot_hits_vs_misses(GMM_results)
print(GMM_results)

# # ====================================== LOF ========================================

lof = LOF(train_pca_features,test_pca_features)
lof.fit_all(n_neighbors=50, algorithm='auto', leaf_size=50, metric='minkowski', p=4, contamination='auto')
LOF_error_prediction = lof.get_batch_predictions(threshold_percentile=10)
LOF_results = evaluation.compare_methods_results(predictions_dict=LOF_error_prediction)
evaluation.plot_hits_vs_misses(LOF_results)
print(LOF_results)

# #===================================STOMPY===========================================
# stmp = MstumpDetector(window_size=200)
# stmp_error_prediction = stmp.get_batch_predictions(test_pca_features, threshold_percentile=95)
# stmp_report = evaluation.compare_methods_results(stmp_error_prediction)
# evaluation.plot_hits_vs_misses(stmp_report)
# print(stmp_report)

# # ================================ LSTMA ===========================================

# lstm_ae = LSTM_AE_Detector(seq_len=250, epochs=10, percentile=95)

# # Fit (na celém slovníku)
# lstm_ae.fit(train_pca_features)

# # Prediction (dostaneš slovník množin)
# ae_outliers = lstm_ae.prediction(test_pca_features)

# # Tvůj stávající report
# report_lstma = evaluation.compare_methods_results(ae_outliers)
# evaluation.plot_hits_vs_misses(report_lstma)
# print(report_lstma)
# #
#=====================================Autoencoder=======================================

# moje_soubory = ['D-13', 'A-6', 'P-2'] 

# # 2. Inicializuj tuner (n_trials=20 je dobrý kompromis rychlost/kvalita)
# tuner = MultiFileSklearnTuner(target_cids=moje_soubory, n_trials=20)

# # 3. Spusť proces (vytvoří sítě, najde parametry a vrátí výsledky)
# ae_results = tuner.tune_and_predict(train_s, test_s, evaluation)

# # 4. Finální report pro ty soubory, které jsi vybral
# report_ae = evaluation.compare_methods_results(ae_results)
# evaluation.plot_hits_vs_misses(report_ae)
# print(report_ae)


