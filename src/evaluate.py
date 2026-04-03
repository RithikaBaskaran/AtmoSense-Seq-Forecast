# Model Evaluation 
import torch
print(torch.cuda.is_available())

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/AQI_Project/data/india_aqi.csv')

!pip install -q torch torchvision pandas numpy scikit-learn matplotlib seaborn kaggle

import torch
CHECKPOINT_PATH = '/content/drive/MyDrive/AQI_Project/checkpoints/best_model.pt'
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {checkpoint.keys()}")
else:
    print(f"Checkpoint is a {type(checkpoint)} object, not a dictionary.")

import sys
import os
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.append('/content/AtmoSense-Seq-Forecast/src')
from model import AQITransformer

# Define paths
CHECKPOINT_PATH = '/content/drive/MyDrive/AQI_Project/checkpoints/best_model.pt'
DATA_PATH = '/content/drive/MyDrive/AQI_Project/data/india_aqi.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {
    'seq_len': 72,
    'pred_len': 48,
    'd_model': 32,
    'nhead': 8,
    'num_enc_layers': 1,
    'num_dec_layers': 1,
    'dim_feedforward': 64,
    'dropout': 0.1,
    'batch_size': 128
}

# 1. Load and Prepare Data Directly
print('Loading data from Drive...')
df = pd.read_csv(DATA_PATH)

feat_cols = [
    'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO (ug/m3)', 'NO2 (ug/m3)',
    'NOx (ppb)', 'NH3 (ug/m3)', 'SO2 (ug/m3)', 'CO (mg/m3)',
    'Ozone (ug/m3)', 'Benzene (ug/m3)', 'Toluene (ug/m3)'
]

print(f"Selected features ({len(feat_cols)}):", feat_cols)

# Clean and scale
data = df[feat_cols].ffill().bfill().values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

test_start = int(len(scaled_data) * 0.8)
test_data = scaled_data[test_start:]

def create_sequences(data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len):
        xs.append(data[i:(i + seq_len)])
        ys.append(data[(i + seq_len):(i + seq_len + pred_len)])
    return np.array(xs), np.array(ys)

X_test, y_test = create_sequences(test_data, params['seq_len'], params['pred_len'])
print(f'Test sequences created: {X_test.shape}')

# 2. Load Model
n_features = len(feat_cols)
model = AQITransformer(
    n_features=n_features, n_targets=n_features, seq_len=params['seq_len'], pred_len=params['pred_len'],
    d_model=params['d_model'], nhead=params['nhead'], num_enc_layers=params['num_enc_layers'],
    num_dec_layers=params['num_dec_layers'], dim_feedforward=params['dim_feedforward'], dropout=params['dropout']
).to(device)

print(f'Loading weights...')
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# 3. Inference Function
@torch.no_grad()
def greedy_decode(model, src, pred_len, device):
    B = src.size(0)
    dec_input = torch.zeros(B, 1, src.size(-1), device=device)
    outputs = []
    for _ in range(pred_len):
        out = model(src, dec_input)
        next_step = out[:, -1:, :]
        outputs.append(next_step)
        dec_input = torch.cat([dec_input, next_step], dim=1)
    return torch.cat(outputs, dim=1)

# 4. Run Inference on a subset
LIMIT = 10
print(f'Running inference on {LIMIT} samples...')
src_subset = torch.tensor(X_test[:LIMIT], dtype=torch.float32).to(device)
tgt_subset = y_test[:LIMIT]

preds_scaled = greedy_decode(model, src_subset, params['pred_len'], device).cpu().numpy()

# 5. Inverse Scaling & Metrics
preds_inv = scaler.inverse_transform(preds_scaled.reshape(-1, n_features)).reshape(LIMIT, params['pred_len'], n_features)
trues_inv = scaler.inverse_transform(tgt_subset.reshape(-1, n_features)).reshape(LIMIT, params['pred_len'], n_features)

results = []
plt.figure(figsize=(12, 18))
for i, name in enumerate(feat_cols):
    p, t = preds_inv[:, :, i], trues_inv[:, :, i]
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t)**2))
    results.append({'Pollutant': name, 'MAE': mae, 'RMSE': rmse})

    plt.subplot(len(feat_cols), 1, i+1)
    plt.plot(t[0, :], label='Actual', color='blue')
    plt.plot(p[0, :], label='Predicted', linestyle='--', color='red')
    plt.title(f'{name} (Sample 0 Forecast)')
    plt.legend()

display(pd.DataFrame(results))
plt.tight_layout()
plt.show()



# Ensure results directory exists
RESULTS_DIR = '/content/AtmoSense-Seq-Forecast/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration for full evaluation
BATCH_SIZE = 64
num_samples = len(X_test)
all_preds = []
all_trues = []

print(f'Starting full evaluation on {num_samples} sequences...')

model.eval()
with torch.no_grad():
    # Using a larger batch size for the source but still decoding one by one
    for i in range(0, num_samples, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, num_samples)
        src_batch = torch.tensor(X_test[i:end_idx], dtype=torch.float32).to(device)
        tgt_batch = y_test[i:end_idx]

        # Greedy decode for the batch
        preds_batch_scaled = greedy_decode(model, src_batch, params['pred_len'], device).cpu().numpy()

        all_preds.append(preds_batch_scaled)
        all_trues.append(tgt_batch)

        if (i // BATCH_SIZE) % 10 == 0:
            print(f'Processed {end_idx}/{num_samples} samples...')

# Concatenate results
all_preds = np.concatenate(all_preds, axis=0)
all_trues = np.concatenate(all_trues, axis=0)

# Inverse scaling
preds_inv = scaler.inverse_transform(all_preds.reshape(-1, n_features)).reshape(-1, params['pred_len'], n_features)
trues_inv = scaler.inverse_transform(all_trues.reshape(-1, n_features)).reshape(-1, params['pred_len'], n_features)

# Calculate Global Metrics
final_results = []
for i, name in enumerate(feat_cols):
    p, t = preds_inv[:, :, i], trues_inv[:, :, i]
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t)**2))
    final_results.append({'Pollutant': name, 'MAE': mae, 'RMSE': rmse})

    # Save visualization for each pollutant (using the first sample as a representative)
    plt.figure(figsize=(10, 4))
    plt.plot(t[0, :], label='Actual', color='blue')
    plt.plot(p[0, :], label='Predicted', linestyle='--', color='red')
    plt.title(f'Full Eval: {name} Forecast')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'forecast_{name.split(" ")[0]}.png'))
    plt.close()

# Save Metrics to CSV
metrics_df = pd.DataFrame(final_results)
metrics_df.to_csv(os.path.join(RESULTS_DIR, 'metrics.csv'), index=False)

print(f'Full evaluation complete. Results saved to {RESULTS_DIR}')
display(metrics_df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define Persistence Baseline
# The 'last' value of each input sequence in X_test is used as the prediction for all 48 steps in the forecast.
# X_test shape: (num_samples, seq_len, n_features)
# Last value shape: (num_samples, n_features)

last_observed = X_test[:, -1, :]
# Broadcast last_observed to match y_test shape (num_samples, pred_len, n_features)
persistence_preds_scaled = np.repeat(last_observed[:, np.newaxis, :], params['pred_len'], axis=1)

# Inverse Scale the Persistence Predictions
persistence_preds_inv = scaler.inverse_transform(persistence_preds_scaled.reshape(-1, n_features)).reshape(-1, params['pred_len'], n_features)
# trues_inv is already available from previous cells

# Calculate Persistence Metrics
baseline_results = []
for i, name in enumerate(feat_cols):
    p, t = persistence_preds_inv[:, :, i], trues_inv[:, :, i]
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t)**2))
    baseline_results.append({'Pollutant': name, 'Baseline_MAE': mae, 'Baseline_RMSE': rmse})

baseline_df = pd.DataFrame(baseline_results)

# Compare with Transformer Model Metrics
comparison_df = pd.merge(metrics_df, baseline_df, on='Pollutant')
comparison_df['MAE_Improvement_%'] = ((comparison_df['Baseline_MAE'] - comparison_df['MAE']) / comparison_df['Baseline_MAE']) * 100

print("Comparison: Transformer Model vs. Persistence Baseline")
display(comparison_df)

# Save comparison
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'baseline_comparison.csv'), index=False)

# Visualize Comparison
plt.figure(figsize=(14, 8))

x = np.arange(len(feat_cols))
width = 0.35

plt.bar(x - width/2, comparison_df['Baseline_MAE'], width, label='Persistence (Baseline)', color='gray', alpha=0.7)
plt.bar(x + width/2, comparison_df['MAE'], width, label='AQITransformer', color='skyblue')

plt.xlabel('Pollutant')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE Comparison: Transformer vs. Persistence Baseline')
plt.xticks(x, [n.split(' ')[0] for n in feat_cols], rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 8))
corr = df[feat_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Pollutants')
heatmap_path = os.path.join(RESULTS_DIR, 'analysis_correlation_heatmap.png')
plt.savefig(heatmap_path)
plt.close()

pollutant_idx = 0
residuals = (all_preds[:, :, pollutant_idx] - all_trues[:, :, pollutant_idx]).flatten()
plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--')
plt.title(f'Prediction Errors (Residuals) - {feat_cols[pollutant_idx]}')
residual_path = os.path.join(RESULTS_DIR, 'analysis_residuals_pm25.png')
plt.savefig(residual_path)
plt.close()


mae_per_step = np.mean(np.abs(all_preds - all_trues), axis=(0, 2))
plt.figure(figsize=(10, 5))
plt.plot(range(1, params['pred_len'] + 1), mae_per_step, marker='o', color='green')
plt.title('MAE vs. Forecast Horizon')
plt.xlabel('Hours into Future')
plt.ylabel('Global MAE')
horizon_path = os.path.join(RESULTS_DIR, 'analysis_mae_horizon.png')
plt.savefig(horizon_path)
plt.close()

# Sync files to Drive
drive_path = '/content/drive/MyDrive/AQI_Project/final_results'
for f in ['analysis_correlation_heatmap.png', 'analysis_residuals_pm25.png', 'analysis_mae_horizon.png']:
    shutil.copy2(os.path.join(RESULTS_DIR, f), os.path.join(drive_path, f))

print(f'Advanced visualizations saved and copied to {drive_path}')

def calculate_mape(y_true, y_pred):
    # Avoid division by zero by adding a small epsilon
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

final_comprehensive_results = []

for i, name in enumerate(feat_cols):
    # Model predictions
    p_model = preds_inv[:, :, i]
    # Baseline (Persistence) predictions
    p_base = persistence_preds_inv[:, :, i]
    # Actual values
    t = trues_inv[:, :, i]

    # Model Metrics
    m_mae = np.mean(np.abs(p_model - t))
    m_rmse = np.sqrt(np.mean((p_model - t)**2))
    m_mape = calculate_mape(t, p_model)

    # Baseline Metrics
    b_mae = np.mean(np.abs(p_base - t))
    b_rmse = np.sqrt(np.mean((p_base - t)**2))
    b_mape = calculate_mape(t, p_base)

    final_comprehensive_results.append({
        'Pollutant': name,
        'Model_MAE': m_mae,
        'Model_RMSE': m_rmse,
        'Model_MAPE_%': m_mape,
        'Baseline_MAE': b_mae,
        'Baseline_RMSE': b_rmse,
        'Baseline_MAPE_%': b_mape
    })

comprehensive_df = pd.DataFrame(final_comprehensive_results)

comprehensive_df.to_csv(os.path.join(RESULTS_DIR, 'metrics_comprehensive.csv'), index=False)
comprehensive_df.to_csv('/content/drive/MyDrive/AQI_Project/final_results/metrics_comprehensive.csv', index=False)

print('Comprehensive metrics including MAPE and Baseline comparison generated.')
display(comprehensive_df)

import seaborn as sns

plt.figure(figsize=(10, 8))
corr = df[feat_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Pollutants')
plt.show()


# Calculating error for all samples for PM2.5 as a representative
pollutant_idx = 0 # PM2.5
residuals = (all_preds[:, :, pollutant_idx] - all_trues[:, :, pollutant_idx]).flatten()

plt.figure(figsize=(10, 5))
sns.histplot(residuals, kde=True, color='purple')
plt.axvline(0, color='red', linestyle='--')
plt.title(f'Distribution of Prediction Errors (Residuals) - {feat_cols[pollutant_idx]}')
plt.xlabel('Error (Predicted - Actual)')
plt.ylabel('Frequency')
plt.show()

mae_per_step = np.mean(np.abs(all_preds - all_trues), axis=(0, 2))

plt.figure(figsize=(10, 5))
plt.plot(range(1, params['pred_len'] + 1), mae_per_step, marker='o', color='green')
plt.title('Mean Absolute Error vs. Forecast Horizon')
plt.xlabel('Hours into Future')
plt.ylabel('Global MAE')
plt.grid(True, alpha=0.3)
plt.show()

print('Visualizations generated. These help identify if the model error accumulates over time.')

import os

results_path = '/content/AtmoSense-Seq-Forecast/results'
if os.path.exists(results_path):
    print(f'Files found in {results_path}:')
    display(os.listdir(results_path))
else:
    print('Results directory not found!')

import shutil

drive_results_path = '/content/drive/MyDrive/AQI_Project/final_results'

os.makedirs(drive_results_path, exist_ok=True)

local_results_path = '/content/AtmoSense-Seq-Forecast/results'
for item in os.listdir(local_results_path):
    s = os.path.join(local_results_path, item)
    d = os.path.join(drive_results_path, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)

print(f'Successfully copied all results to: {drive_results_path}')