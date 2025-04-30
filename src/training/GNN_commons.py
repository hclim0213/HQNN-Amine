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

def load_model(file_path, model_name=None):
    models_dict = torch.load(file_path)
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
        for smile, y in zip(df.RDKit_SMILES, df.y):
            mol = Chem.MolFromSmiles(smile)
            featurizer = CanonicalFeaturizer()
            data = featurizer.process(mol, y)
            datas.append(data)
    elif type == 2:
        for smile, y, temp in zip(df.RDKit_SMILES, df.y, df.temp):
            mol = Chem.MolFromSmiles(smile)
            featurizer = CanonicalFeaturizer()
            data = featurizer.process(mol, y, temp)
            datas.append(data)
    return datas

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return {'r2': r2, 'mse': mse, 'mae': mae, 'rmse': rmse}

def evaluate(model, dataloader, loss_func, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            target = data.y.to(device)
            
            data = data.to(device)
            output = model(data)
            
            output = output.view(-1)
            target = target.float().view(-1)
            
            total_loss += loss_func(output, target).item()
            all_preds.append(output.cpu().numpy().reshape(-1,1))
            all_targets.append(target.cpu().numpy().reshape(-1,1))

    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()
    if np.isnan(all_preds).any():
        print("Warnings: NaNs detected in predictions")
    all_preds = np.nan_to_num(all_preds, nan=0.0)
    
    metrics = calculate_metrics(all_targets, all_preds)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics
    
def train(model, train_dataloader, val_dataloader, optimizer, loss_func, epochs, device, model_name, cv):
    best_score = -float('inf')
    best_epoch = -1
    best_train_metrics = None
    best_val_metrics = None

    for epoch in range(epochs):
        model.train()
        for data in train_dataloader:
            data = data.to(device)
            target = data.y.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)

            output = output.view(-1)
            target = target.float().view(-1)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_dataloader, loss_func, device)
        val_metrics = evaluate(model, val_dataloader, loss_func, device)

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