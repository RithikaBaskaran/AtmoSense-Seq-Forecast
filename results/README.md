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
| Persistence Baseline | 16.05 | 43.23 | 0.6047 | -- |
| AQITransformer | 12.80 | 33.41 | 0.7639 | 22.71% |
| HybridBiLSTMTransformer | 12.96 | 33.19 | 0.7670 | 23.23% |

## Results Summary

Both Transformer-based models outperform the persistence baseline on the full held-out test set.

The **HybridBiLSTMTransformer** achieves the best overall RMSE and R²:

- RMSE: 33.19
- R²: 0.7670
- RMSE improvement over baseline: 23.23%

The **AQITransformer** achieves slightly better MAE:

- MAE: 12.80
- RMSE: 33.41
- RMSE improvement over baseline: 22.71%

Overall, the HybridBiLSTMTransformer is selected as the best model by RMSE, while AQITransformer remains competitive and performs slightly better by average absolute error.

### Overall RMSE vs Persistence Baseline

<p align="center">
  <img src="figures/figures/overall_rmse_model_vs_baseline.png" alt="Overall RMSE vs Persistence Baseline" width="600">
</p>

This figure shows that both AQITransformer and HybridBiLSTMTransformer reduce RMSE substantially compared with the persistence baseline.

### Per-Pollutant RMSE Comparison

<p align="center">
  <img src="figures/figures/per_pollutant_rmse_aqi_vs_hybrid.png" alt="Per-Pollutant RMSE Comparison" width="700">
</p>

This figure compares AQITransformer and HybridBiLSTMTransformer across individual pollutants. The hybrid model performs slightly better overall, although the gains vary by pollutant.

## Files

### Metrics

- `overall_aqi_vs_hybrid_comparison.csv`  
  Overall MAE, RMSE, R², and baseline-improvement comparison between AQITransformer and HybridBiLSTMTransformer.

- `overall_r2_aqi_vs_hybrid.csv`  
  Overall R² comparison between AQITransformer, HybridBiLSTMTransformer, and the persistence baseline.

- `per_pollutant_aqi_vs_hybrid_comparison.csv`  
  Per-pollutant comparison between both architectures.

- `metrics_full_test.csv`  
  Full-test AQITransformer metrics.

- `metrics_hybrid_full_test.csv`  
  Full-test HybridBiLSTMTransformer metrics.

### Figures

- `overall_rmse_model_vs_baseline.png`  
  Overall RMSE comparison against the persistence baseline.

- `overall_rmse_improvement_percent.png`  
  Overall RMSE improvement percentage over the persistence baseline.

- `per_pollutant_rmse_aqi_vs_hybrid.png`  
  Per-pollutant RMSE comparison between AQITransformer and HybridBiLSTMTransformer.

- `hybrid_vs_aqi_rmse_improvement_by_pollutant.png`  
  Hybrid model RMSE improvement over AQITransformer by pollutant.

- `rmse_by_forecast_horizon_aqi_vs_hybrid.png`  
  RMSE across the 48-hour forecast horizon.

- `mae_by_forecast_horizon_aqi_vs_hybrid.png`  
  MAE across the 48-hour forecast horizon.

- `pollutant_correlation_heatmap.png`  
  Pearson correlation heatmap showing relationships between pollutant channels.
