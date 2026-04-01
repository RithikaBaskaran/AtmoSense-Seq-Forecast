import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/content/AtmoSense-Seq-Forecast/src')

from dataset import build_dataloaders
from model import AQITransformer


def get_args():
    p = argparse.ArgumentParser(description='Train AQI Seq2Seq Transformer')

    p.add_argument('--data_dir',       type=str,
                   default='/content/drive/MyDrive/AQI_Project/data')
    p.add_argument('--checkpoint_dir', type=str,
                   default='/content/drive/MyDrive/AQI_Project/checkpoints')
    p.add_argument('--log_path',       type=str,
                   default='/content/drive/MyDrive/AQI_Project/experiment_logs/training_log.txt')

    p.add_argument('--seq_len',  type=int, default=72)
    p.add_argument('--pred_len', type=int, default=48)

    p.add_argument('--d_model',         type=int,   default=128)
    p.add_argument('--nhead',           type=int,   default=8)
    p.add_argument('--num_enc_layers',  type=int,   default=4)
    p.add_argument('--num_dec_layers',  type=int,   default=4)
    p.add_argument('--dim_feedforward', type=int,   default=512)
    p.add_argument('--dropout',         type=float, default=0.1)

    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--patience',     type=int,   default=5)
    p.add_argument('--tf_ratio',     type=float, default=0.5)

    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed',        type=int, default=42)

    return p.parse_args()


@torch.no_grad()
def greedy_decode(model, src, pred_len, n_targets, device):
    B         = src.size(0)
    dec_input = torch.zeros(B, 1, n_targets, device=device)
    outputs   = []

    for _ in range(pred_len):
        out       = model(src, dec_input)
        next_step = out[:, -1:, :]
        outputs.append(next_step)
        dec_input = torch.cat([dec_input, next_step], dim=1)

    return torch.cat(outputs, dim=1)


def compute_metrics(pred, true, eps=1e-6):
    mae  = torch.mean(torch.abs(pred - true)).item()
    rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()
    mape = torch.mean(torch.abs((pred - true) / (true.abs() + eps))).item() * 100
    return mae, rmse, mape


def train_one_epoch(model, loader, optimizer, criterion, device,
                    pred_len, n_targets, tf_ratio):
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        B = src.size(0)
        optimizer.zero_grad()

        if torch.rand(1).item() < tf_ratio:
            start  = torch.zeros(B, 1, n_targets, device=device)
            tgt_in = torch.cat([start, tgt[:, :-1, :]], dim=1)
            output = model(src, tgt_in)
        else:
            dec_in = torch.zeros(B, 1, n_targets, device=device)
            steps  = []
            for _ in range(pred_len):
                out    = model(src, dec_in)
                step   = out[:, -1:, :]
                steps.append(step)
                dec_in = torch.cat([dec_in, step.detach()], dim=1)
            output = torch.cat(steps, dim=1)

        loss = criterion(output, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device, pred_len, n_targets):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        output   = greedy_decode(model, src, pred_len, n_targets, device)
        loss     = criterion(output, tgt)
        total_loss += loss.item()
        all_preds.append(output.cpu())
        all_trues.append(tgt.cpu())

    preds = torch.cat(all_preds)
    trues = torch.cat(all_trues)
    mae, rmse, mape = compute_metrics(preds, trues)

    return {'loss': total_loss / len(loader),
            'mae': mae, 'rmse': rmse, 'mape': mape}


def log(msg, path):
    print(msg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        f.write(msg + '\n')


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='steelblue')
    plt.plot(val_losses,   label='Val Loss',   color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('AtmoSense — Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Loss curve saved → {save_path}')


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*65}')
    print(f'  AtmoSense-Seq-Forecast | Training | Device: {device}')
    print(f'{"="*65}\n')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    loss_curve_path = os.path.join(args.checkpoint_dir, 'loss_curve.png')

    print('Building dataloaders...')
    train_loader, val_loader, test_loader, scaler, feat_cols = build_dataloaders(
        data_dir        = args.data_dir,
        seq_len         = args.seq_len,
        pred_len        = args.pred_len,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        scaler_save_dir = args.checkpoint_dir,
    )

    sample_x, sample_y = next(iter(train_loader))
    n_features = sample_x.shape[-1]
    n_targets  = sample_y.shape[-1]

    print(f'  n_features : {n_features}')
    print(f'  n_targets  : {n_targets}')
    print(f'  feat_cols  : {feat_cols}')
    print(f'  train batches : {len(train_loader):,}')
    print(f'  val   batches : {len(val_loader):,}\n')

    model = AQITransformer(
        n_features      = n_features,
        n_targets       = n_targets,
        seq_len         = args.seq_len,
        pred_len        = args.pred_len,
        d_model         = args.d_model,
        nhead           = args.nhead,
        num_enc_layers  = args.num_enc_layers,
        num_dec_layers  = args.num_dec_layers,
        dim_feedforward = args.dim_feedforward,
        dropout         = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}\n')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    no_improve    = 0
    train_losses, val_losses = [], []

    header = (f'{"Ep":>4} | {"TrainLoss":>10} | {"ValLoss":>10} | '
              f'{"MAE":>8} | {"RMSE":>8} | {"MAPE%":>7} | Time')
    log('\n' + header, args.log_path)
    log('-' * len(header), args.log_path)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            pred_len=args.pred_len, n_targets=n_targets, tf_ratio=args.tf_ratio,
        )
        val_m = evaluate(
            model, val_loader, criterion, device,
            pred_len=args.pred_len, n_targets=n_targets,
        )
        val_loss = val_m['loss']
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        row = (f'{epoch:>4} | {train_loss:>10.6f} | {val_loss:>10.6f} | '
               f'{val_m["mae"]:>8.4f} | {val_m["rmse"]:>8.4f} | '
               f'{val_m["mape"]:>6.2f}% | {time.time()-t0:.1f}s')
        log(row, args.log_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), checkpoint_path)
            log(f'       ✓ Best checkpoint saved (val {val_loss:.6f})', args.log_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log(f'\nEarly stopping at epoch {epoch}.', args.log_path)
                break

    log(f'\nBest val loss : {best_val_loss:.6f}', args.log_path)
    log(f'Checkpoint    : {checkpoint_path}',     args.log_path)
    plot_losses(train_losses, val_losses, loss_curve_path)


if __name__ == '__main__':
    main()
