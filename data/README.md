# Data and Checkpoint Setup

This file explains how to set up the dataset, trained model checkpoints, scalers, and cached evaluation outputs required to reproduce the final results.

The actual data and model artifacts are stored in Google Drive because they are too large to commit to GitHub.

## Google Drive Folder

Use the shared project folder:

https://drive.google.com/drive/folders/1FoA62gJLeSTH15AsRnVHR7pYGsUivXuP?usp=sharing

To reproduce the results:

1. Open the Google Drive link.
2. Copy or add the folder to your own Google Drive.
3. Open the evaluation notebook in Google Colab.
4. Mount Google Drive.
5. Make sure the folder is available at:

```text
/content/drive/MyDrive/AQI_Project/
```

The evaluation notebook and scripts expect this path by default.

## Drive Structure

```text
AQI_Project/
├── data/
├── checkpoints/
│   ├── best_model.pt
│   ├── all_scaler.pkl
│   ├── best_model_hybrid.pt
│   └── all_scaler_hybrid.pkl
└── Final Full Set Evaluation Results/
```

## Dataset

This project uses hourly air quality monitoring data from India’s Central Pollution Control Board (CPCB), sourced through the Kaggle dataset:

**Time Series Air Quality Data of India (2010–2023)**

The dataset contains station-level hourly air quality readings across India. The model uses 11 pollutant columns:

```text
PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3, Benzene, Toluene
```

Weather and non-pollutant columns are excluded from model inputs.

## Preprocessing Summary

The preprocessing pipeline:

1. Loads station-level CPCB CSV files.
2. Filters out stations with high missing-data rates.
3. Retains the 11 pollutant columns used for forecasting.
4. Performs interpolation independently within each station.
5. Splits the data chronologically into train, validation, and test sets.
6. Fits a `StandardScaler` on the training set only.
7. Builds sliding-window samples:
   - input window: 72 hours
   - forecast horizon: 48 hours
   - stride: 6 hours

The model predicts all 11 pollutants jointly for the next 48 hours.

## Checkpoint Files

The `checkpoints/` folder contains the trained model weights and fitted scalers.

### `best_model.pt`

Checkpoint for the original `AQITransformer`.

### `all_scaler.pkl`

Saved `StandardScaler` used with the `AQITransformer`.

### `best_model_hybrid.pt`

Checkpoint for the `HybridBiLSTMTransformer`.

### `all_scaler_hybrid.pkl`

Saved `StandardScaler` used with the `HybridBiLSTMTransformer`.

## Cached Evaluation Results

The folder:

```text
Final Full Set Evaluation Results/
```

contains cached full-test prediction arrays and saved evaluation outputs.

These cached files allow the evaluation notebook to regenerate metrics and plots without rerunning full model inference.

The full evaluation uses:

```text
test batches: 2,332
test samples: 149,225
input window: 72 hours
forecast horizon: 48 hours
pollutants: 11
```

## Reproduction Modes

The evaluation notebook supports two modes.

### Cached Evaluation Mode

Recommended for quick reproduction:

```python
RUN_FULL_AQI_EVAL = False
RUN_FULL_HYBRID_EVAL = False
```

This loads cached prediction arrays from Google Drive and regenerates tables and plots.

### Full Inference Mode

Use this to rerun full model evaluation:

```python
RUN_FULL_AQI_EVAL = True
RUN_FULL_HYBRID_EVAL = True
```

This loads trained checkpoints, runs inference on the full held-out test set, recomputes metrics, and overwrites cached results in Google Drive.
