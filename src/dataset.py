"""
dataset.py — Person 1 deliverable
India AQI Transformer Project

Multi-target forecasting: predicts ALL 7 pollutants simultaneously.
Window sizes: 72-hour input (seq_len) → 48-hour output (pred_len).

Exports:
    AQIDataset        : PyTorch Dataset (per-station sliding window, multi-target)
    build_dataloaders : full pipeline data_dir → (train_loader, val_loader, test_loader, ...)

Usage (Person 3 in train.py):
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/AQI_Project')
    from dataset import build_dataloaders

    DATA_DIR = '/content/drive/MyDrive/AQI_Project/data'
    train_loader, val_loader, test_loader, scaler, feat_cols = build_dataloaders(DATA_DIR)
    # x: (batch, 72, n_features)   y: (batch, 48, n_targets)
    # scaler.inverse_transform(arr_2d)  → real-unit values for all pollutants
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ── Column mapping from raw CSV headers to clean names ────────────────────────
COL_RENAME = {
    'From Date':           'date',
    'PM2.5 (ug/m3)':      'PM2.5',
    'PM10 (ug/m3)':       'PM10',
    'NO (ug/m3)':         'NO',
    'NO2 (ug/m3)':        'NO2',
    'NOx (ppb)':          'NOx',
    'NH3 (ug/m3)':        'NH3',
    'SO2 (ug/m3)':        'SO2',
    'CO (mg/m3)':         'CO',
    'Ozone (ug/m3)':      'O3',
    'Benzene (ug/m3)':    'Benzene',
    'Toluene (ug/m3)':    'Toluene',
}

STATION_COL    = 'StationId'
DATE_COL       = 'date'
POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                  'NH3', 'SO2', 'CO', 'O3', 'Benzene', 'Toluene']

STATION_MISSING_THRESHOLD = 0.40
COL_MISSING_THRESHOLD     = 0.50
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15


# ── Data loading & cleaning ───────────────────────────────────────────────────

def _load_and_clean(data_dir: str):
    """
    Read all per-station CSVs, merge, clean, and return a DataFrame plus
    the list of surviving pollutant column names (used as both input and output).
    """
    station_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f != 'stations_info.csv'
    ])

    frames = []
    for fname in station_files:
        station_id = fname.replace('.csv', '')
        sdf = pd.read_csv(
            os.path.join(data_dir, fname),
            parse_dates=['From Date'],
        )
        rename = {k: v for k, v in COL_RENAME.items() if k in sdf.columns}
        sdf = sdf[list(rename.keys())].rename(columns=rename)
        sdf[STATION_COL] = station_id
        frames.append(sdf)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values([STATION_COL, DATE_COL]).reset_index(drop=True)

    # Remove stations with too much missing data
    present = [c for c in POLLUTANT_COLS if c in df.columns]
    station_missing = (
        df.groupby(STATION_COL)[present]
        .apply(lambda g: g.isnull().mean().mean())
    )
    good = station_missing[station_missing <= STATION_MISSING_THRESHOLD].index
    df   = df[df[STATION_COL].isin(good)].copy()

    # Drop sparse pollutant columns (>50% missing globally)
    col_miss  = df[present].isnull().mean()
    bad_cols  = col_miss[col_miss > COL_MISSING_THRESHOLD].index.tolist()
    df        = df.drop(columns=bad_cols)
    # All surviving columns are both model inputs AND outputs (multi-target)
    all_cols  = [c for c in present if c not in bad_cols]

    # Per-station interpolation (never across station boundaries)
    df[all_cols] = (
        df.groupby(STATION_COL)[all_cols]
        .transform(lambda g: g.interpolate(method='linear', limit_direction='both'))
    )
    df[all_cols] = (
        df.groupby(STATION_COL)[all_cols]
        .transform(lambda g: g.ffill().bfill())
    )

    df = df.dropna(subset=all_cols).reset_index(drop=True)
    return df, all_cols


def _split_by_time(df, train_frac, val_frac):
    dates        = df[DATE_COL].sort_values()
    train_cutoff = dates.quantile(train_frac)
    val_cutoff   = dates.quantile(train_frac + val_frac)
    train = df[df[DATE_COL] <= train_cutoff].copy()
    val   = df[(df[DATE_COL] > train_cutoff) & (df[DATE_COL] <= val_cutoff)].copy()
    test  = df[df[DATE_COL] > val_cutoff].copy()
    return train, val, test


def _scale(train, val, test, all_cols):
    """Single StandardScaler covering all pollutant columns (inputs = outputs)."""
    scaler = StandardScaler()
    train, val, test = train.copy(), val.copy(), test.copy()
    train[all_cols] = scaler.fit_transform(train[all_cols])
    val[all_cols]   = scaler.transform(val[all_cols])
    test[all_cols]  = scaler.transform(test[all_cols])
    return train, val, test, scaler


# ── Dataset ───────────────────────────────────────────────────────────────────

class AQIDataset(Dataset):
    """
    Per-station sliding-window Dataset for multi-pollutant forecasting.
    No window ever spans two different monitoring stations.

    Returns per __getitem__:
        x : FloatTensor  (seq_len,  n_features)  — past 72 hrs, all pollutants
        y : FloatTensor  (pred_len, n_targets)   — next 48 hrs, all pollutants
    """

    def __init__(self, df, station_col, feature_cols, target_cols,
                 seq_len=72, pred_len=48):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self._feat    = {}
        self._tgt     = {}
        self._indices = []

        for station, grp in df.groupby(station_col):
            grp = grp.sort_values(DATE_COL).reset_index(drop=True)
            self._feat[station] = torch.FloatTensor(grp[feature_cols].values)
            self._tgt[station]  = torch.FloatTensor(grp[target_cols].values)
            n = len(grp)
            for i in range(n - seq_len - pred_len + 1):
                self._indices.append((station, i))

        if not self._indices:
            raise ValueError(
                f'No valid windows (seq_len={seq_len}, pred_len={pred_len}).'
            )

    def __len__(self):  return len(self._indices)

    def __getitem__(self, idx):
        station, i = self._indices[idx]
        x = self._feat[station][i : i + self.seq_len]
        y = self._tgt[station][i + self.seq_len : i + self.seq_len + self.pred_len]
        return x, y

    @property
    def n_features(self):  return next(iter(self._feat.values())).shape[1]

    @property
    def n_targets(self):   return next(iter(self._tgt.values())).shape[1]

    @property
    def n_stations(self):  return len(self._feat)


# ── Public API ────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    seq_len: int         = 72,
    pred_len: int        = 48,
    batch_size: int      = 64,
    num_workers: int     = 2,
    scaler_save_dir: str = None,
):
    """
    Full pipeline: folder of station CSVs → (train_loader, val_loader, test_loader).

    Args:
        data_dir        : path to the folder containing per-station CSV files
        seq_len         : look-back window in hours  (default 72)
        pred_len        : forecast horizon in hours  (default 48)
        batch_size      : DataLoader batch size      (default 64)
        num_workers     : DataLoader worker processes (default 2)
        scaler_save_dir : if set, saves all_scaler.pkl here

    Returns:
        train_loader, val_loader, test_loader, scaler, feat_cols
        Tensor shapes — x: (batch, seq_len, n_features)
                        y: (batch, pred_len, n_targets)
    """
    df, all_cols = _load_and_clean(data_dir)
    train_df, val_df, test_df = _split_by_time(df, TRAIN_FRAC, VAL_FRAC)
    train_df, val_df, test_df, scaler = _scale(train_df, val_df, test_df, all_cols)

    if scaler_save_dir:
        os.makedirs(scaler_save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(scaler_save_dir, 'all_scaler.pkl'))

    ds_kw = dict(station_col=STATION_COL, feature_cols=all_cols,
                 target_cols=all_cols, seq_len=seq_len, pred_len=pred_len)
    dl_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return (
        DataLoader(AQIDataset(train_df, **ds_kw), shuffle=True,  **dl_kw),
        DataLoader(AQIDataset(val_df,   **ds_kw), shuffle=False, **dl_kw),
        DataLoader(AQIDataset(test_df,  **ds_kw), shuffle=False, **dl_kw),
        scaler, all_cols,
    )
