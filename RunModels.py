import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from Dataset import DatasetOperations
from TimeContext import TimeContextModif
from Evaluation import EvaluateResults
from LSTMAutoencoder import LSTM_AE_Detector
# from KMeans import KMeansAnalyzer
# from DBSCANModel import DBSCANModel
# from IsolationForestModel import IsolationForestModel
# from OneClassSVMModel import OneClassSVMModel

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
train_path       = os.path.join(BASE_DIR, "archive", "data", "data", "train")
test_path        = os.path.join(BASE_DIR, "archive", "data", "data", "test")
result_file_path = os.path.join(BASE_DIR, "archive", "labeled_anomalies.csv")

dt = DatasetOperations(test_path=test_path, train_path=train_path, results_path=result_file_path)
train_dataset, test_dataset, results = dt.load_data()

train_subset, test_subset = dt.select_subset(random_selection=True, manual_file_names=None, subset_size=10, seed=42)

evaluation = EvaluateResults()
evaluation.load_solution(results)

# ─────────────────────────────────────────────
# 2. Feature engineering (rolling statistics)
# ─────────────────────────────────────────────
tc = TimeContextModif(train_dataset=train_subset, test_dataset=test_subset)
train_fe, test_fe = tc.add_rolling_statistics(window_length=200)

# ─────────────────────────────────────────────
# 3. K-Means (clustering + feature enrichment)
# ─────────────────────────────────────────────
# print("\n" + "="*60)
# print("K-MEANS")
# print("="*60)
# km = KMeansAnalyzer(train_fe, test_fe)
# km.fit_all(k=5)
# train_km, test_km = km.get_enriched_features()
# km_preds   = km.get_batch_predictions(threshold_percentile=95)
# km_results = evaluation.compare_methods_results(predictions_dict=km_preds)
# evaluation.plot_hits_vs_misses(km_results)
# print(km_results)

# ─────────────────────────────────────────────
# 4. DBSCAN
# ─────────────────────────────────────────────
# print("\n" + "="*60)
# print("DBSCAN — PARAMETER SWEEP")
# print("="*60)
# dbscan_summary = []
# for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
#     db = DBSCANModel(train_km, test_km)
#     db.fit_all(eps=eps, min_samples=10)
#     preds = db.get_batch_predictions()
#     report = evaluation.compare_methods_results(predictions_dict=preds)
#     macro_f1 = report['F1_Score'].mean() if not report.empty else 0.0
#     dbscan_summary.append({'eps': eps, 'Macro F1': round(macro_f1, 4)})
#     print(f"  eps={eps:.1f}  →  Macro F1: {macro_f1:.4f}")
# best_eps = max(dbscan_summary, key=lambda x: x['Macro F1'])['eps']
# print(f"\nBest eps: {best_eps}  — running final DBSCAN with plots")
# db_best = DBSCANModel(train_km, test_km)
# db_best.fit_all(eps=best_eps, min_samples=10)
# db_preds   = db_best.get_batch_predictions()
# db_results = evaluation.compare_methods_results(predictions_dict=db_preds)
# evaluation.plot_hits_vs_misses(db_results)
# print(db_results)

# ─────────────────────────────────────────────
# 5. Isolation Forest
# ─────────────────────────────────────────────
# print("\n" + "="*60)
# print("ISOLATION FOREST — PARAMETER SWEEP")
# print("="*60)
# iso_summary = []
# for contamination, percentile in [(0.05, None), (0.10, None), (0.05, 5), (0.05, 10)]:
#     iso = IsolationForestModel(train_km, test_km)
#     iso.fit_all(contamination=contamination)
#     preds = iso.get_batch_predictions(threshold_percentile=percentile)
#     report = evaluation.compare_methods_results(predictions_dict=preds)
#     macro_f1 = report['F1_Score'].mean() if not report.empty else 0.0
#     label = f"contam={contamination}, percentile={percentile}"
#     iso_summary.append({'config': label, 'Macro F1': round(macro_f1, 4)})
#     print(f"  {label}  →  Macro F1: {macro_f1:.4f}")
# best_config = max(iso_summary, key=lambda x: x['Macro F1'])
# iso_best = IsolationForestModel(train_km, test_km)
# iso_best.fit_all(contamination=float(best_config['config'].split(',')[0].split('=')[1]))
# iso_preds   = iso_best.get_batch_predictions()
# iso_results = evaluation.compare_methods_results(predictions_dict=iso_preds)
# evaluation.plot_hits_vs_misses(iso_results)
# print(iso_results)

# ─────────────────────────────────────────────
# 6. One-Class SVM
# ─────────────────────────────────────────────
# print("\n" + "="*60)
# print("ONE-CLASS SVM")
# print("="*60)
# svm = OneClassSVMModel(train_km, test_km)
# svm.fit_all(nu=0.05)
# svm_preds   = svm.get_batch_predictions()
# svm_results = evaluation.compare_methods_results(predictions_dict=svm_preds)
# evaluation.plot_hits_vs_misses(svm_results)
# print(svm_results)

# ─────────────────────────────────────────────
# 7. LSTM Autoencoder
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("LSTM AUTOENCODER")
print("="*60)
lstm = LSTM_AE_Detector(seq_len=50, hidden_dim=32, epochs=5, percentile=99, n_features=1)
lstm.fit(train_subset)
lstm_preds   = lstm.prediction(test_subset)
lstm_results = evaluation.compare_methods_results(predictions_dict=lstm_preds)
evaluation.plot_hits_vs_misses(lstm_results, title='LSTM Autoencoder')
print(lstm_results)
