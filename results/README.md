# Evaluation Results

This folder contains the final evaluation outputs for **AtmoSense-Seq-Forecast**.

The final evaluation compares:

- `AQITransformer`
- `HybridBiLSTMTransformer`
- `Persistence baseline`

## Evaluation Setup

- Input window: 72 hours
- Forecast horizon: 48 hours
- Dataset stride: 6
- Pollutants: 11
- Test batches: 2,332
- Test samples: 149,225
- Baseline: Persistence baseline, where the last observed value from the input window is repeated across all 48 forecast hours.

## Main Results

| Model | MAE | RMSE | R² | RMSE Improvement vs Baseline |
|---|---:|---:|---:|---:|
| AQITransformer | 12.80 | 33.41 | 0.7639 | 22.71% |
| HybridBiLSTMTransformer | 12.96 | 33.19 | 0.7670 | 23.23% |
| Persistence Baseline | 16.05 | 43.23 | 0.6047 | -- |

The HybridBiLSTMTransformer achieved the best overall RMSE and R², while AQITransformer achieved slightly better MAE.

## Files

### Metrics

- `overall_aqi_vs_hybrid_comparison.csv`  
  Overall comparison between AQITransformer and HybridBiLSTMTransformer.

- `per_pollutant_aqi_vs_hybrid_comparison.csv`  
  Per-pollutant comparison between both architectures.

- `metrics_full_test.csv`  
  Full-test AQITransformer metrics.

- `metrics_hybrid_full_test.csv`  
  Full-test HybridBiLSTMTransformer metrics.

### Figures

- `overall_rmse_model_vs_baseline.png`  
  Overall RMSE comparison against persistence baseline.

- `overall_rmse_improvement_percent.png`  
  Overall RMSE improvement percentage over persistence baseline.

- `per_pollutant_rmse_aqi_vs_hybrid.png`  
  Per-pollutant RMSE comparison between AQITransformer and HybridBiLSTMTransformer.

- `hybrid_vs_aqi_rmse_improvement_by_pollutant.png`  
  Hybrid model RMSE improvement over AQITransformer by pollutant.

- `rmse_by_forecast_horizon_aqi_vs_hybrid.png`  
  RMSE across the 48-hour forecast horizon.

- `mae_by_forecast_horizon_aqi_vs_hybrid.png`  
  MAE across the 48-hour forecast horizon.
