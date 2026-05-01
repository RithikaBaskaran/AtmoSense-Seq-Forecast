 


import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ── Column mapping: raw CSV headers → clean internal names ───────────────────
# These match exactly the headers found in the per-station CSV files
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
# All other columns in the CSV (To Date, RH, WS, WD, SR, BP, VWS, AT, RF,
# Eth-Benzene, MP-Xylene, O Xylene) are intentionally ignored.

STATION_COL    = 'StationId'
DATE_COL       = 'date'
POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                  'NH3', 'SO2', 'CO', 'O3', 'Benzene', 'Toluene']

STATION_MISSING_THRESHOLD = 0.40   # drop stations with >40% missing
COL_MISSING_THRESHOLD     = 0.50   # drop pollutant columns with >50% missing
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15


# ── Data loading & cleaning ───────────────────────────────────────────────────

def _load_and_clean(data_dir: str):
    """
    Read all per-station CSVs from data_dir.
    Keeps only columns in COL_RENAME — all weather columns are dropped.
    Returns a merged DataFrame and the list of surviving pollutant column names.
    """
    station_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.csv') and f != 'stations_info.csv'
                              and f != 'india_aqi.csv'   # ignore merged file if present
    ])

    if not station_files:
        raise FileNotFoundError(
            f'No per-station CSV files found in {data_dir}. '
            f'Expected files like AP002.csv, not a single merged CSV.'
        )

    frames = []
    for fname in station_files:
        station_id = fname.replace('.csv', '')
        sdf = pd.read_csv(
            os.path.join(data_dir, fname),
            parse_dates=['From Date'],
        )

        # Keep ONLY columns that are in COL_RENAME — drops weather, xylene, etc.
        rename = {k: v for k, v in COL_RENAME.items() if k in sdf.columns}
        if not rename:
            print(f'  Warning: no matching columns in {fname}, skipping.')
            continue

        sdf = sdf[list(rename.keys())].rename(columns=rename)
        sdf[STATION_COL] = station_id
        frames.append(sdf)

    if not frames:
        raise ValueError('No valid station files could be loaded.')

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values([STATION_COL, DATE_COL]).reset_index(drop=True)

    # ── Station-level quality filter ─────────────────────────────────────────
    present = [c for c in POLLUTANT_COLS if c in df.columns]
    station_missing = (
        df.groupby(STATION_COL)[present]
          .apply(lambda g: g.isnull().mean().mean())
    )
    good = station_missing[station_missing <= STATION_MISSING_THRESHOLD].index
    df   = df[df[STATION_COL].isin(good)].copy()
    print(f'  Stations retained: {len(good)} of {station_missing.shape[0]}')

    # ── Column-level quality filter ──────────────────────────────────────────
    col_miss = df[present].isnull().mean()
    bad_cols = col_miss[col_miss > COL_MISSING_THRESHOLD].index.tolist()
    df       = df.drop(columns=bad_cols)
    all_cols = [c for c in present if c not in bad_cols]
    print(f'  Pollutant columns retained: {len(all_cols)} → {all_cols}')

    # ── Per-station interpolation ────────────────────────────────────────────
    df[all_cols] = (
        df.groupby(STATION_COL)[all_cols]
          .transform(lambda g: g.interpolate(method='linear',
                                             limit_direction='both'))
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
    val   = df[(df[DATE_COL] > train_cutoff)
               & (df[DATE_COL] <= val_cutoff)].copy()
    test  = df[df[DATE_COL] > val_cutoff].copy()
    return train, val, test


def _scale(train, val, test, all_cols):
    """Single StandardScaler across all pollutant columns."""
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
                 seq_len=72, pred_len=48, stride=6):
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.stride   = stride
        self._feat    = {}
        self._tgt     = {}
        self._indices = []

        for station, grp in df.groupby(station_col):
            grp = grp.sort_values(DATE_COL).reset_index(drop=True)
            self._feat[station] = torch.FloatTensor(grp[feature_cols].values)
            self._tgt[station]  = torch.FloatTensor(grp[target_cols].values)
            n = len(grp)
            for i in range(0, n - seq_len - pred_len + 1, stride):
                self._indices.append((station, i))

        if not self._indices:
            raise ValueError(
                f'No valid windows found (seq_len={seq_len}, '
                f'pred_len={pred_len}). Check station data lengths.'
            )

    def __len__(self):  return len(self._indices)

    def __getitem__(self, idx):
        station, i = self._indices[idx]
        x = self._feat[station][i : i + self.seq_len]
        y = self._tgt[station][i + self.seq_len
                                : i + self.seq_len + self.pred_len]
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
    stride=6
):
    """
    Full pipeline: folder of per-station CSVs → train/val/test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, scaler, feat_cols
        Tensor shapes — x: (batch, seq_len, n_features)
                        y: (batch, pred_len, n_targets)
        Both n_features and n_targets will equal the number of pollutant
        columns that survive quality filtering (typically 7).
    """
    print('Loading and cleaning data...')
    df, all_cols = _load_and_clean(data_dir)

    train_df, val_df, test_df = _split_by_time(df, TRAIN_FRAC, VAL_FRAC)
    train_df, val_df, test_df, scaler = _scale(
        train_df, val_df, test_df, all_cols
    )

    if scaler_save_dir:
        os.makedirs(scaler_save_dir, exist_ok=True)
        joblib.dump(scaler,
                    os.path.join(scaler_save_dir, 'all_scaler.pkl'))
        print(f'  Scaler saved → {scaler_save_dir}/all_scaler.pkl')

    ds_kw = dict(station_col=STATION_COL, feature_cols=all_cols,
                 target_cols=all_cols, seq_len=seq_len, pred_len=pred_len, stride=stride)
    dl_kw = dict(batch_size=batch_size,
                 num_workers=num_workers, pin_memory=True)

    train_ds = AQIDataset(train_df, **ds_kw)
    val_ds   = AQIDataset(val_df,   **ds_kw)
    test_ds  = AQIDataset(test_df,  **ds_kw)

    print(f'  Training windows   : {len(train_ds):,}')
    print(f'  Validation windows : {len(val_ds):,}')
    print(f'  Test windows       : {len(test_ds):,}')
    print(f'  n_features = n_targets = {train_ds.n_features}')

    return (
        DataLoader(train_ds, shuffle=True,  **dl_kw),
        DataLoader(val_ds,   shuffle=False, **dl_kw),
        DataLoader(test_ds,  shuffle=False, **dl_kw),
        scaler, all_cols,
    )
