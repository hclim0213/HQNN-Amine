import os
import time
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .utility import create_param_dir_name
from itertools import product


LOGGER = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'r2': r2, 'mse': mse, 'mae': mae, 'rmse': rmse}

def evaluate(model, dataloader, loss_func, device, scaler):
    model.eval()
    total_loss = 0.0
    preds, original_targets = [], []

    with torch.no_grad():
        for data, target, original_target in dataloader:
            # Move everything to the device and convert targets to float
            data = data.to(device)
            target = target.to(device).float().view(-1)
            original_target = original_target.to(device).float().view(-1)
            
            # Forward pass and compute loss
            output = model(data).view(-1)
            total_loss += loss_func(output, target).item()
            
            # Save predictions and original targets as numpy arrays (reshaped for scaler)
            preds.append(output.cpu().numpy().reshape(-1, 1))
            original_targets.append(original_target.cpu().numpy().reshape(-1, 1))

    # Concatenate batch results
    preds_array = np.concatenate(preds)
    if np.isnan(preds_array).any():
        print("Warning: NaNs detected in predictions")
    preds_array = np.nan_to_num(preds_array, nan=0.0)
    
    # Inverse-transform predictions
    preds_scaled = scaler.inverse_transform(preds_array).ravel()
    original_targets_array = np.concatenate(original_targets).ravel()
    
    metrics = calculate_metrics(original_targets_array, preds_scaled)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_func, num_epochs, device, save_dir, scaler):
    best_model = None
    best_score = -float('inf')
    best_epoch = 0
    best_train_metrics = None
    best_val_metrics = None
    
    metrics_dict = {
        'train_r2': [], 'val_r2': [],
        'train_mae': [], 'val_mae': [],
        'train_mse': [], 'val_mse': [],
        'train_rmse': [], 'val_rmse': [],
        'train_loss': [], 'val_loss': []
    }
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        train_start_time = time.time()
        model.train()
        train_loss =0 
        for data, target, _ in train_dataloader:
            data, target = data.to(device), target.to(device)
            print(f"Target: {target.numpy()}, Original Target: {_.numpy()}")

            optimizer.zero_grad(set_to_none=True)

            output = model(data)
            print(f"Model Output: {output.view(-1).detach().cpu().numpy()}")

            output = output.view(-1)
            target = target.float().view(-1)

            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time

        # Evaluation phase - Train data
        eval_train_start_time = time.time()
        train_metrics = evaluate(model, train_dataloader, loss_func, device, scaler)
        eval_train_end_time = time.time()
        eval_train_duration = eval_train_end_time - eval_train_start_time

        # Evaluation phase - Validation data
        eval_val_start_time = time.time()
        val_metrics = evaluate(model, val_dataloader, loss_func, device, scaler)
        eval_val_end_time = time.time()
        eval_val_duration = eval_val_end_time - eval_val_start_time
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        
        metrics_dict['train_r2'].append(train_metrics['r2'])
        metrics_dict['val_r2'].append(val_metrics['r2'])
        metrics_dict['train_mae'].append(train_metrics['mae'])
        metrics_dict['val_mae'].append(val_metrics['mae'])
        metrics_dict['train_mse'].append(train_metrics['mse'])
        metrics_dict['val_mse'].append(val_metrics['mse'])
        metrics_dict['train_rmse'].append(train_metrics['rmse'])
        metrics_dict['val_rmse'].append(val_metrics['rmse'])
        metrics_dict['train_loss'].append(train_loss / len(train_dataloader))
        metrics_dict['val_loss'].append(val_metrics['loss'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Duration (Total/Train/Train Eval/Val Eval): {epoch_duration:.2f}/{train_duration:.2f}/{eval_train_duration:.2f}/{eval_val_duration:.2f} secs")
        print(f"Train - R2: {train_metrics['r2']:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        print(f"Valid - R2: {val_metrics['r2']:.4f}, MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
                
        # Save latest model
        torch.save(model.state_dict(), os.path.join(save_dir, 'last_model.pth'))

        # Check if this is the best model so far
        current_score = max(0,train_metrics['r2']) * max(0,val_metrics['r2'])

        if current_score > best_score:
            best_score = current_score
            best_model = model.state_dict()
            best_epoch = epoch + 1
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            torch.save(best_model, os.path.join(save_dir, 'best_model.pth'))
            
            with open(os.path.join(save_dir, 'best_model_info.txt'), 'w') as f:
                f.write(f"Best Epoch: {best_epoch}\n")
                f.write(f"Best Score (train_r2*_val_r2): {best_score:.6f}\n")
                f.write(f"Best Epoch Total Duration: {epoch_duration:.2f} seconds\n")
                f.write(f"Best Epoch Training Duration: {train_duration:.2f} seconds\n")
                f.write(f"Best Epoch Train Evaluation Duration: {eval_train_duration:.2f} seconds\n")
                f.write(f"Best Epoch Validation Evaluation Duration: {eval_val_duration:.2f} seconds\n")
                f.write("\nBest Train Metrics:\n")
                for k, v in best_train_metrics.items():
                    f.write(f"{k}: {v:.6f}\n")
                f.write("\nBest Validation Metrics:\n")
                for k, v in best_val_metrics.items():
                    f.write(f"{k}: {v:.6f}\n")

    return best_train_metrics, best_val_metrics, best_score, best_epoch, metrics_dict


def perform_mlp(
        X, y, y_scaled, ps, param_grid, device, save_dir, scaler,
        model_class=None,
        custom_dataset=None,
        base_filename="dataset",
        pretrained_models=None,
        fold_to_run=None,
):
    """
    Grid-search for a *classical* (or hybrid) model.
    Only the original five hyper-parameters are assumed:
        hidden_sizes • batch_size • learning_rate • num_epochs • dropout_rate
    """
    from .qnn import Classical as _DefaultModel, CustomDataset as _DefaultDS
    from .qnn import Hybrid as _Hybrid

    model_class      = model_class      or _DefaultModel
    custom_dataset   = custom_dataset   or _DefaultDS
    best_params      = None
    best_score_sofar = -float("inf")
    results          = []

    for combo in product(*param_grid.values()):
        param_dict   = dict(zip(param_grid.keys(), combo))
        param_dir    = create_param_dir_name(param_dict)
        param_path   = os.path.join(save_dir, param_dir)
        os.makedirs(param_path, exist_ok=True)
        fold_metrics = []

        # decide which folds to run
        splits = list(ps.split())
        fold_enumerator = (
            [(fold_to_run, splits[fold_to_run])]
            if fold_to_run is not None else
            list(enumerate(splits))
        )

        for fold_idx, (tr_idx, val_idx) in fold_enumerator:
            # ─ datasets & loaders
            ds_train = custom_dataset(X.iloc[tr_idx],  y.iloc[tr_idx],
                                      y_scaled.iloc[tr_idx])
            ds_val   = custom_dataset(X.iloc[val_idx], y.iloc[val_idx],
                                      y_scaled.iloc[val_idx])
            dl_train = DataLoader(ds_train, batch_size=param_dict["batch_size"],
                                  shuffle=True)
            dl_val   = DataLoader(ds_val,   batch_size=param_dict["batch_size"],
                                  shuffle=False)

            # ─ model
            if issubclass(model_class, _Hybrid):
                model = model_class(
                    X.shape[1],
                    param_dict["hidden_sizes"],
                    1,
                    param_dict["dropout_rate"],
                    param_dict["n_qubit"],
                    param_dict["n_layer"],
                    param_dict["device_type"],
                    pretrained_classical = (
                        pretrained_models[fold_idx] if pretrained_models else None
                    ),
                )
            else:  # plain MLP
                model = model_class(
                    X.shape[1],
                    param_dict["hidden_sizes"],
                    1,
                    param_dict["dropout_rate"],
                )
            model.to(device)

            optim  = torch.optim.Adam(
                        model.parameters(),
                        lr = param_dict["learning_rate"]  # ← NO weight-decay
                     )
            lossfn = torch.nn.MSELoss()

            fold_dir = os.path.join(param_path, f"fold_{fold_idx+1}")
            os.makedirs(fold_dir, exist_ok=True)

            tr_m, va_m, score, best_ep, log = train_and_evaluate(
                model, dl_train, dl_val,
                optim, lossfn,
                num_epochs = param_dict["num_epochs"],
                device     = device,
                save_dir   = fold_dir,
                scaler     = scaler,
            )
            fold_metrics.append((tr_m, va_m, score))

        # consolidate fold results
        mean_score = np.mean([
            max(0, tr["r2"]) * max(0, va["r2"])
            for tr, va, _ in fold_metrics
        ])
        results.append({"params": param_dict, "score": mean_score})

        if mean_score > best_score_sofar:
            best_score_sofar = mean_score
            best_params      = param_dict


    return best_params, results, best_score_sofar
