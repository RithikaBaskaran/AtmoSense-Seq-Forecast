"""
Full Evaluation for AQITransformer and HybridBiLSTMTransformer.
"""

import os, sys, math, json, joblib, importlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

REPO_DIR = '/content/AtmoSense-Seq-Forecast'
SRC_DIR = os.path.join(REPO_DIR, 'src')
sys.path.insert(0, SRC_DIR)

from dataset import build_dataloaders

import model as model_module
importlib.reload(model_module)
from model import AQITransformer, HybridBiLSTMTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Paths
DATA_DIR = "/content/drive/MyDrive/AQI_Project/data"
CHECKPOINT_DIR = "/content/drive/MyDrive/AQI_Project/checkpoints"

AQI_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
AQI_SCALER_PATH = os.path.join(CHECKPOINT_DIR, "all_scaler.pkl")

HYBRID_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model_hybrid.pt")
HYBRID_SCALER_PATH = os.path.join(CHECKPOINT_DIR, "all_scaler_hybrid.pkl")

LOCAL_RESULTS_DIR = os.path.join(REPO_DIR, "final_full_set_eval_results")
DRIVE_RESULTS_DIR = "/content/drive/MyDrive/AQI_Project/Final Full Set Evaluation Results"

os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)

# Model/data constants
SEQ_LEN = 72
PRED_LEN = 48
BATCH_SIZE = 64
DATASET_STRIDE = 6

# Run mode
# False = load saved arrays from Drive and regenerate tables/figures.
# True  = rerun full inference and overwrite arrays/metrics with the same run names.
RUN_FULL_AQI_EVAL = False
RUN_FULL_HYBRID_EVAL = False

AQI_RUN_NAME = "full_test"
HYBRID_RUN_NAME = "hybrid_full_test"

print("Drive results folder:", DRIVE_RESULTS_DIR)
print("Run AQI full eval:", RUN_FULL_AQI_EVAL)
print("Run Hybrid full eval:", RUN_FULL_HYBRID_EVAL)

"""Build held-out test loader"""

train_loader, val_loader, test_loader, loader_scaler, feat_cols = build_dataloaders(
    DATA_DIR,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    stride=DATASET_STRIDE,
    batch_size=BATCH_SIZE,
    num_workers=0,
    scaler_save_dir=None,
)

n_features = len(feat_cols)
n_targets = len(feat_cols)

print("Pollutants:", feat_cols)
print("n_features:", n_features)
print("n_targets:", n_targets)
print("test batches:", len(test_loader))
print("test samples:", len(test_loader.dataset))
print("dataset stride:", DATASET_STRIDE)


def greedy_decode(model, src, pred_len, n_targets, device):
    model.eval()
    batch_size = src.size(0)
    decoder_input = torch.zeros(batch_size, 1, n_targets, device=device)

    preds = []
    with torch.no_grad():
        for _ in range(pred_len):
            out = model(src, decoder_input)
            next_step = out[:, -1:, :]
            preds.append(next_step)
            decoder_input = torch.cat([decoder_input, next_step], dim=1)

    return torch.cat(preds, dim=1)


def inverse_scale_3d(arr, scaler, n_features):
    original_shape = arr.shape
    return scaler.inverse_transform(arr.reshape(-1, n_features)).reshape(original_shape)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

def save_arrays(run_name, preds_scaled, trues_scaled, baseline_scaled, preds, trues, baseline):
    for folder in [LOCAL_RESULTS_DIR, DRIVE_RESULTS_DIR]:
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, f"preds_scaled_{run_name}.npy"), preds_scaled)
        np.save(os.path.join(folder, f"trues_scaled_{run_name}.npy"), trues_scaled)
        np.save(os.path.join(folder, f"baseline_scaled_{run_name}.npy"), baseline_scaled)
        np.save(os.path.join(folder, f"preds_{run_name}.npy"), preds)
        np.save(os.path.join(folder, f"trues_{run_name}.npy"), trues)
        np.save(os.path.join(folder, f"baseline_{run_name}.npy"), baseline)


def load_arrays(run_name, results_dir):
    preds = np.load(os.path.join(results_dir, f"preds_{run_name}.npy"))
    trues = np.load(os.path.join(results_dir, f"trues_{run_name}.npy"))
    baseline = np.load(os.path.join(results_dir, f"baseline_{run_name}.npy"))
    return preds, trues, baseline


def find_cached_dir(run_name, preferred_dir, fallback_dirs=None):
    fallback_dirs = fallback_dirs or []
    candidates = [preferred_dir] + fallback_dirs
    for folder in candidates:
        if os.path.exists(os.path.join(folder, f"preds_{run_name}.npy")):
            return folder
    raise FileNotFoundError(
        f"Could not find cached arrays for run_name='{run_name}'. "
        f"Set the matching RUN_FULL_*_EVAL flag to True to regenerate them."
    )

"""Model loading"""

AQI_SCALER = joblib.load(AQI_SCALER_PATH)
HYBRID_SCALER = joblib.load(HYBRID_SCALER_PATH)

print("AQI scaler features:", AQI_SCALER.n_features_in_)
print("Hybrid scaler features:", HYBRID_SCALER.n_features_in_)
print("n_features:", n_features)

assert AQI_SCALER.n_features_in_ == n_features, "AQI scaler feature mismatch"
assert HYBRID_SCALER.n_features_in_ == n_features, "Hybrid scaler feature mismatch"

AQI_MODEL_CONFIG = dict(
    n_features=n_features,
    n_targets=n_targets,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    d_model=128,
    nhead=8,
    num_enc_layers=4,
    num_dec_layers=4,
    dim_feedforward=512,
    dropout=0.1,
)

HYBRID_MODEL_CONFIG = dict(
    n_features=n_features,
    n_targets=n_targets,
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    d_model=128,
    nhead=8,
    num_dec_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    lstm_layers=2,
)


def load_state_dict_from_checkpoint(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    print("Checkpoint:", checkpoint_path)
    print("Number of keys:", len(state.keys()))
    print("First keys:", list(state.keys())[:10])
    print("Has BiLSTM keys:", any("bilstm" in k for k in state.keys()))
    print("Has Transformer encoder keys:", any("encoder.layers" in k for k in state.keys()))
    return state


def load_aqi_model(checkpoint_path):
    print("=" * 70)
    print("Loading AQITransformer")
    model_obj = AQITransformer(**AQI_MODEL_CONFIG).to(device)
    state = load_state_dict_from_checkpoint(checkpoint_path)
    model_obj.load_state_dict(state)
    model_obj.eval()
    return model_obj


def load_hybrid_model(checkpoint_path):
    print("=" * 70)
    print("Loading HybridBiLSTMTransformer")
    model_obj = HybridBiLSTMTransformer(**HYBRID_MODEL_CONFIG).to(device)
    state = load_state_dict_from_checkpoint(checkpoint_path)
    model_obj.load_state_dict(state)
    model_obj.eval()
    return model_obj

"""
Full-test evaluation function

If full evaluation is enabled, rerunning full evaluation will overwrite the matching cached arrays and metric files in Drive.
"""

def run_full_test_evaluation(model_obj, scaler_obj, run_name):
    all_preds_scaled = []
    all_trues_scaled = []
    all_baseline_scaled = []

    used = 0
    model_obj.eval()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            preds_batch = greedy_decode(model_obj, x, PRED_LEN, n_targets, device)
            baseline_batch = x[:, -1:, :].repeat(1, PRED_LEN, 1)

            all_preds_scaled.append(preds_batch.cpu().numpy())
            all_trues_scaled.append(y.cpu().numpy())
            all_baseline_scaled.append(baseline_batch.cpu().numpy())

            used += 1
            if used % 50 == 0:
                print(f"{run_name}: Finished {used}/{len(test_loader)} batches")

    preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    trues_scaled = np.concatenate(all_trues_scaled, axis=0)
    baseline_scaled = np.concatenate(all_baseline_scaled, axis=0)

    preds = inverse_scale_3d(preds_scaled, scaler_obj, n_features)
    trues = inverse_scale_3d(trues_scaled, scaler_obj, n_features)
    baseline = inverse_scale_3d(baseline_scaled, scaler_obj, n_features)

    save_arrays(run_name, preds_scaled, trues_scaled, baseline_scaled, preds, trues, baseline)

    print("=" * 70)
    print("Run:", run_name)
    print("Evaluated batches:", used)
    print("Evaluated samples:", preds.shape[0])
    print("preds:", preds.shape)
    print("trues:", trues.shape)
    print("baseline:", baseline.shape)
    print("Saved arrays to:", DRIVE_RESULTS_DIR)

    return preds, trues, baseline, used

"""Metrics functions"""

def make_metrics_table(run_name, model_label, preds_arr, trues_arr, baseline_arr, evaluated_batches):
    rows = []

    for i, pollutant in enumerate(feat_cols):
        y_true = trues_arr[:, :, i]
        y_model = preds_arr[:, :, i]
        y_base = baseline_arr[:, :, i]

        model_mae = mae(y_true, y_model)
        base_mae = mae(y_true, y_base)
        model_rmse = rmse(y_true, y_model)
        base_rmse = rmse(y_true, y_base)

        rows.append({
            "model": model_label,
            "pollutant": pollutant,
            "model_mae": model_mae,
            "model_rmse": model_rmse,
            "model_mape": mape(y_true, y_model),
            "model_r2": r2_score_np(y_true, y_model),
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse,
            "baseline_mape": mape(y_true, y_base),
            "baseline_r2": r2_score_np(y_true, y_base),
            "mae_improvement_percent": ((base_mae - model_mae) / base_mae) * 100 if base_mae != 0 else np.nan,
            "rmse_improvement_percent": ((base_rmse - model_rmse) / base_rmse) * 100 if base_rmse != 0 else np.nan,
            "evaluated_batches": evaluated_batches,
            "evaluated_samples": preds_arr.shape[0],
            "run_name": run_name,
        })

    metrics_df = pd.DataFrame(rows)

    for folder in [LOCAL_RESULTS_DIR, DRIVE_RESULTS_DIR]:
        os.makedirs(folder, exist_ok=True)
        metrics_df.to_csv(os.path.join(folder, f"metrics_{run_name}.csv"), index=False)

    return metrics_df


def overall_metrics(run_name, model_label, preds_arr, trues_arr, baseline_arr, evaluated_batches):
    model_mae = mae(trues_arr, preds_arr)
    base_mae = mae(trues_arr, baseline_arr)
    model_rmse = rmse(trues_arr, preds_arr)
    base_rmse = rmse(trues_arr, baseline_arr)

    return {
        "model": model_label,
        "run_name": run_name,
        "model_mae": model_mae,
        "baseline_mae": base_mae,
        "mae_improvement_percent": ((base_mae - model_mae) / base_mae) * 100 if base_mae != 0 else np.nan,
        "model_rmse": model_rmse,
        "baseline_rmse": base_rmse,
        "rmse_improvement_percent": ((base_rmse - model_rmse) / base_rmse) * 100 if base_rmse != 0 else np.nan,
        "model_r2": r2_score_np(trues_arr, preds_arr),
        "baseline_r2": r2_score_np(trues_arr, baseline_arr),
        "evaluated_batches": evaluated_batches,
        "evaluated_samples": preds_arr.shape[0],
    }

"""Evaluate or load AQITransformer results"""

if RUN_FULL_AQI_EVAL:
    AQI_MODEL = load_aqi_model(AQI_CHECKPOINT_PATH)
    aqi_preds, aqi_trues, aqi_baseline, aqi_batches = run_full_test_evaluation(
        AQI_MODEL,
        AQI_SCALER,
        AQI_RUN_NAME,
    )
else:
    AQI_RESULTS_DIR = find_cached_dir(
        AQI_RUN_NAME,
        DRIVE_RESULTS_DIR,
        fallback_dirs=["/content/drive/MyDrive/AQI_Project/results3"],
    )
    aqi_preds, aqi_trues, aqi_baseline = load_arrays(AQI_RUN_NAME, AQI_RESULTS_DIR)
    aqi_batches = len(test_loader)
    print("Loaded AQI arrays from:", AQI_RESULTS_DIR)

metrics_aqi_full = make_metrics_table(
    AQI_RUN_NAME,
    "AQITransformer",
    aqi_preds,
    aqi_trues,
    aqi_baseline,
    aqi_batches,
)

overall_aqi = overall_metrics(
    AQI_RUN_NAME,
    "AQITransformer",
    aqi_preds,
    aqi_trues,
    aqi_baseline,
    aqi_batches,
)

print("AQI preds:", aqi_preds.shape)
display(metrics_aqi_full)
display(pd.DataFrame([overall_aqi]))

"""Evaluate or load HybridBiLSTMTransformer results"""

if RUN_FULL_HYBRID_EVAL:
    HYBRID_MODEL = load_hybrid_model(HYBRID_CHECKPOINT_PATH)
    hybrid_preds, hybrid_trues, hybrid_baseline, hybrid_batches = run_full_test_evaluation(
        HYBRID_MODEL,
        HYBRID_SCALER,
        HYBRID_RUN_NAME,
    )
else:
    HYBRID_RESULTS_DIR = find_cached_dir(HYBRID_RUN_NAME, DRIVE_RESULTS_DIR)
    hybrid_preds, hybrid_trues, hybrid_baseline = load_arrays(HYBRID_RUN_NAME, HYBRID_RESULTS_DIR)
    hybrid_batches = len(test_loader)
    print("Loaded Hybrid arrays from:", HYBRID_RESULTS_DIR)

metrics_hybrid_full = make_metrics_table(
    HYBRID_RUN_NAME,
    "HybridBiLSTMTransformer",
    hybrid_preds,
    hybrid_trues,
    hybrid_baseline,
    hybrid_batches,
)

overall_hybrid = overall_metrics(
    HYBRID_RUN_NAME,
    "HybridBiLSTMTransformer",
    hybrid_preds,
    hybrid_trues,
    hybrid_baseline,
    hybrid_batches,
)

print("Hybrid preds:", hybrid_preds.shape)
display(metrics_hybrid_full)
display(pd.DataFrame([overall_hybrid]))

"""Compare AQITransformer and HybridBiLSTMTransformer"""

overall_compare = pd.DataFrame([overall_aqi, overall_hybrid])
display(overall_compare)

for folder in [LOCAL_RESULTS_DIR, DRIVE_RESULTS_DIR]:
    overall_compare.to_csv(os.path.join(folder, "overall_aqi_vs_hybrid_comparison.csv"), index=False)

per_pollutant_compare = metrics_aqi_full.merge(
    metrics_hybrid_full,
    on="pollutant",
    suffixes=("_aqi", "_hybrid"),
)

per_pollutant_compare["hybrid_vs_aqi_rmse_improvement_percent"] = (
    (per_pollutant_compare["model_rmse_aqi"] - per_pollutant_compare["model_rmse_hybrid"])
    / per_pollutant_compare["model_rmse_aqi"]
) * 100

per_pollutant_compare["hybrid_vs_aqi_mae_improvement_percent"] = (
    (per_pollutant_compare["model_mae_aqi"] - per_pollutant_compare["model_mae_hybrid"])
    / per_pollutant_compare["model_mae_aqi"]
) * 100

cols_to_show = [
    "pollutant",
    "model_rmse_aqi",
    "model_rmse_hybrid",
    "hybrid_vs_aqi_rmse_improvement_percent",
    "model_mae_aqi",
    "model_mae_hybrid",
    "hybrid_vs_aqi_mae_improvement_percent",
]

display(per_pollutant_compare[cols_to_show])

for folder in [LOCAL_RESULTS_DIR, DRIVE_RESULTS_DIR]:
    per_pollutant_compare.to_csv(os.path.join(folder, "per_pollutant_aqi_vs_hybrid_comparison.csv"), index=False)

"""Visualization"""

def save_current_plot(filename):
    for folder in [LOCAL_RESULTS_DIR, DRIVE_RESULTS_DIR]:
        path = os.path.join(folder, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print("Saved:", path)

# 1. Overall RMSE comparison against baseline
plot_df = overall_compare.copy()
x = np.arange(len(plot_df))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, plot_df["baseline_rmse"], width, label="Persistence Baseline")
plt.bar(x + width/2, plot_df["model_rmse"], width, label="Model")
plt.xticks(x, plot_df["model"], rotation=15, ha="right")
plt.ylabel("RMSE")
plt.title("Overall RMSE vs Persistence Baseline")
plt.legend()
plt.tight_layout()
save_current_plot("overall_rmse_model_vs_baseline.png")
plt.show()

# 2. Compact overall RMSE improvement over persistence baseline
plot_df = overall_compare.copy()
plt.figure(figsize=(6, 2.8))
bars = plt.barh(plot_df["model"], plot_df["rmse_improvement_percent"])
plt.xlabel("RMSE Improvement over Persistence (%)")
plt.title("Overall RMSE Improvement")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%", va="center")
plt.xlim(0, plot_df["rmse_improvement_percent"].max() + 3)
plt.tight_layout()
save_current_plot("overall_rmse_improvement_percent.png")
plt.show()

# 3. Per-pollutant RMSE comparison
plot_df = per_pollutant_compare.copy()
x = np.arange(len(plot_df))
width = 0.35

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, plot_df["model_rmse_aqi"], width, label="AQITransformer")
plt.bar(x + width/2, plot_df["model_rmse_hybrid"], width, label="HybridBiLSTMTransformer")
plt.xticks(x, plot_df["pollutant"], rotation=45, ha="right")
plt.ylabel("RMSE")
plt.title("Per-Pollutant RMSE")
plt.legend()
plt.tight_layout()
save_current_plot("per_pollutant_rmse_aqi_vs_hybrid.png")
plt.show()

# 4. Hybrid improvement over AQI by pollutant
plot_df = per_pollutant_compare.sort_values("hybrid_vs_aqi_rmse_improvement_percent", ascending=False)

plt.figure(figsize=(12, 5))
plt.bar(plot_df["pollutant"], plot_df["hybrid_vs_aqi_rmse_improvement_percent"])
plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Hybrid RMSE Improvement over AQI (%)")
plt.title("Hybrid vs AQITransformer: RMSE Improvement by Pollutant")
plt.tight_layout()
save_current_plot("hybrid_vs_aqi_rmse_improvement_by_pollutant.png")
plt.show()

# 5. RMSE across forecast horizon
horizon_rmse_aqi = np.sqrt(np.mean((aqi_trues - aqi_preds) ** 2, axis=(0, 2)))
horizon_rmse_hybrid = np.sqrt(np.mean((hybrid_trues - hybrid_preds) ** 2, axis=(0, 2)))
horizon_rmse_baseline = np.sqrt(np.mean((aqi_trues - aqi_baseline) ** 2, axis=(0, 2)))

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, PRED_LEN + 1), horizon_rmse_baseline, marker="o", label="Persistence Baseline")
plt.plot(np.arange(1, PRED_LEN + 1), horizon_rmse_aqi, marker="o", label="AQITransformer")
plt.plot(np.arange(1, PRED_LEN + 1), horizon_rmse_hybrid, marker="o", label="HybridBiLSTMTransformer")
plt.xlabel("Forecast Hour")
plt.ylabel("RMSE")
plt.title("RMSE Across 48-Hour Forecast Horizon")
plt.legend()
plt.tight_layout()
save_current_plot("rmse_by_forecast_horizon_aqi_vs_hybrid.png")
plt.show()

# 6. MAE across forecast horizon
horizon_mae_aqi = np.mean(np.abs(aqi_trues - aqi_preds), axis=(0, 2))
horizon_mae_hybrid = np.mean(np.abs(hybrid_trues - hybrid_preds), axis=(0, 2))
horizon_mae_baseline = np.mean(np.abs(aqi_trues - aqi_baseline), axis=(0, 2))

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, PRED_LEN + 1), horizon_mae_baseline, marker="o", label="Persistence Baseline")
plt.plot(np.arange(1, PRED_LEN + 1), horizon_mae_aqi, marker="o", label="AQITransformer")
plt.plot(np.arange(1, PRED_LEN + 1), horizon_mae_hybrid, marker="o", label="HybridBiLSTMTransformer")
plt.xlabel("Forecast Hour")
plt.ylabel("MAE")
plt.title("MAE Across 48-Hour Forecast Horizon")
plt.legend()
plt.tight_layout()
save_current_plot("mae_by_forecast_horizon_aqi_vs_hybrid.png")
plt.show()

"""Summary"""

print("=" * 70)
print("TEST SUMMARY")
print("Test batches:", len(test_loader))
print("Test samples:", len(test_loader.dataset))
print("Pollutants:", feat_cols)
print()

print("Overall comparison:")
display(overall_compare)

best_model = overall_compare.sort_values("model_rmse").iloc[0]
print("Best model by overall RMSE:", best_model["model"])
print("Best overall RMSE:", best_model["model_rmse"])
print("RMSE improvement over persistence:", best_model["rmse_improvement_percent"], "%")
print("Saved all results to:", DRIVE_RESULTS_DIR)
