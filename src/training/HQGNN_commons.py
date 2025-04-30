import numpy as np
import pandas as pd
import torch

from src.utils.featurizer import CanonicalFeaturizer

import logging
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import random
import math

import warnings
warnings.filterwarnings('ignore')

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

import numpy as np

DEVICE_CONFIGS = {
    'brussels': {
        'two_qubit_error': 3.12e-3,
        'median_sx_error': 2.371e-4,
        'readout_error': 1.820e-2,
        't1': 296.65e-6,
        't2': 166.28e-6,
        'delta_t': 660e-9,
    },
    'strasbourg': {
        'two_qubit_error': 2.86e-3,
        'median_sx_error': 2.439e-4,
        'readout_error': 1.810e-2,
        't1': 292.21e-6,
        't2': 191.5e-6,
        'delta_t': 660e-9,
    },
    'fez': {
        'two_qubit_error': 2.792e-3,
        'median_sx_error': 2.703e-4,
        'readout_error': 1.645e-2,
        't1': 118.06e-6,
        't2': 91.41e-6,
        'delta_t': 68e-9,
    },
}

def get_device_metadata(hw_device: str):
    if hw_device not in DEVICE_CONFIGS:
        raise ValueError(f"Unknown device: {hw_device}")

    cfg = DEVICE_CONFIGS[hw_device]
    amplitude_damping_prob = 1 - np.exp(-cfg['delta_t'] / cfg['t1'])
    phase_damping_prob = 1 - np.exp(-cfg['delta_t'] / cfg['t2'])

    return {
        'two_qubit_depolarizing_error': cfg['two_qubit_error'],
        'single_qubit_depolarizing_error': cfg['median_sx_error'],
        'amplitude_damping_prob': amplitude_damping_prob,
        'phase_damping_prob': phase_damping_prob,
        'readout_prob': cfg['readout_error'],
    }

def save_model(model_name, model, cv):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    save_dir = os.path.join(base_dir, 'saved_models')

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{model_name}.pt')

    if os.path.exists(file_path):
        models_dict = torch.load(file_path)
    else:
        models_dict = {}

    models_dict[f"{model_name}_{cv}"] = model.state_dict()
    torch.save(models_dict, file_path)

def load_model(file_path, device, model_name=None):
    models_dict = torch.load(file_path, map_location=device)
    if model_name is not None:
        if model_name in models_dict:
            return models_dict[model_name]
        else:
            raise KeyError(f"Model '{model_name}' not found in the file.")
    else:
        return models_dict

def smiles2tensor(df, type):
    datas = []
    if type == 1:
        for smile, y_scaled, y in zip(df.RDKit_SMILES, df.y_scaled, df.y):
            mol = Chem.MolFromSmiles(smile)
            featurizer = CanonicalFeaturizer()
            data = featurizer.process(sample=mol, score_scaled=y_scaled, score=y)
            datas.append(data)
    elif type == 2:
        for smile, y_scaled, y, temp in zip(df.RDKit_SMILES, df.y_scaled, df.y, df.temp):
            mol = Chem.MolFromSmiles(smile)
            featurizer = CanonicalFeaturizer()
            data = featurizer.process(sample=mol, score=y, temp=temp, score_scaled=y_scaled)
            datas.append(data)
    return datas

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {'r2': r2, 'mse': mse, 'mae': mae, 'rmse': rmse}

def evaluate(model, dataloader, loss_func, device, scaler):
    model.eval()
    all_preds = []
    all_targets = []
    all_original_targets = []
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            target = data.y_scaled.to(device)
            original_target = data.y.to(device)
            
            data = data.to(device)
            output = model(data)
            
            output = output.view(-1)
            target = target.float().view(-1)
            original_target = original_target.float().view(-1)
            
            total_loss += loss_func(output, target).item()
            all_preds.append(output.cpu().numpy().reshape(-1,1))
            all_targets.append(target.cpu().numpy().reshape(-1,1))
            all_original_targets.append(original_target.cpu().numpy().reshape(-1,1))

    all_preds_scaled_up = scaler.inverse_transform(np.concatenate(all_preds)).ravel()
    all_original_targets = np.concatenate(all_original_targets).ravel()
    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()
    if np.isnan(all_preds).any():
        print("Warnings: NaNs detected in predictions")
    all_preds = np.nan_to_num(all_preds, nan=0.0)
    
    metrics = calculate_metrics(all_original_targets, all_preds_scaled_up)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics

def train(model, train_dataloader, val_dataloader, optimizer, loss_func, epochs, device, model_name, cv, scaler):
    best_score = -float('inf')
    best_epoch = -1
    best_train_metrics = None
    best_val_metrics = None

    for epoch in range(epochs):
        model.train()
        for data in train_dataloader:
            data = data.to(device)
            target = data.y_scaled.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)

            output = output.view(-1)
            target = target.float().view(-1)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_dataloader, loss_func, device, scaler)
        val_metrics = evaluate(model, val_dataloader, loss_func, device, scaler)

        string_result = f"CV\t{cv}\tEpoch\t{epoch}\tTrain_R2\t{train_metrics['r2']}\tTrain_MAE\t{train_metrics['mae']}\t"
        string_result += f"Train_RMSE\t{train_metrics['rmse']}\tValid_R2\t{val_metrics['r2']}\tValid_MAE\t{val_metrics['mae']}\t"
        string_result += f"Valid_MAE\t{val_metrics['mae']}\tValid_MSE\t{val_metrics['mse']}\tValid_RMSE\t{val_metrics['rmse']}"
        print(string_result)

        current_score = max(0, train_metrics['r2']) * max(0, val_metrics['r2'])
        if current_score > best_score:
            best_score = current_score
            best_model_state = model.state_dict()
            best_train_metrics = train_metrics 
            best_epoch = epoch
            best_val_metrics = val_metrics 

    save_model(model_name, model, cv)
    model.load_state_dict(best_model_state)
    return best_train_metrics, best_val_metrics, best_epoch