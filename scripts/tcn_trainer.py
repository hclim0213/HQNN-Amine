import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch

from src.models.GNN import TransformerConvNetwork
from sklearn.model_selection import PredefinedSplit
import argparse
import random

from src.training.GNN_commons import smiles2tensor, train
from torch_geometric import loader
import torch.optim as optim
import torch.nn as nn


import warnings
warnings.filterwarnings('ignore')

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def train_result(df_copy, ps, param_dict, device):

    for cv, (train_idx, valid_idx) in enumerate(ps.split()):
        train_data = smiles2tensor(df_copy.iloc[train_idx], param_dict['type'])
        valid_data = smiles2tensor(df_copy.iloc[valid_idx], param_dict['type'])

        train_dataloader = loader.DataLoader(train_data, batch_size=param_dict['batch_size'], shuffle=True)
        valid_dataloader = loader.DataLoader(valid_data, batch_size=param_dict['batch_size'], shuffle=True)
        
        model = TransformerConvNetwork(input_size=param_dict['input_size'], hidden_size=param_dict['hidden_size'], output_size=param_dict['output_size'], 
                                        num_layers=param_dict['num_layers'], num_heads=param_dict['num_heads'], dropout=param_dict['dropout'], type=param_dict['type'])
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=param_dict['lr'])
        loss_func = nn.MSELoss()

        best_train_metrics, best_val_metrics, best_epoch = train(
            model, train_dataloader, valid_dataloader, optimizer, loss_func, 
            param_dict['epochs'], device, param_dict['model_name'], cv
        )

        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tTrain\tR2\t{best_train_metrics['r2']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tTrain\tRMSE\t{best_train_metrics['rmse']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tTrain\tMAE\t{best_train_metrics['mae']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tTrain\tMSE\t{best_train_metrics['mse']}")
        
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tValid\tR2\t{best_val_metrics['r2']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tValid\tRMSE\t{best_val_metrics['rmse']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tValid\tMAE\t{best_val_metrics['mae']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tValid\tMSE\t{best_val_metrics['mse']}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='TCN training')
    parser.add_argument('--data_file', help='directory')

    args = parser.parse_args()
    data_file = args.data_file
    df = pd.read_csv(data_file, sep='\t')    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    param_dict = {
        'input_size' : 84,
        'output_size' : 1, 
        'dropout' : 0.1, 
        'edge_size' : 12,
        'batch_size' : 128, 
        'epochs' : 1, 
        'lr' : 1e-3, 
    }

    data_type = data_file.split('/')[1].split('.')[0]
    
    type_dict = {
        'pKa'            : 1, 
        'viscosity'      : 2, 
        'melting_point'  : 1,
        'boiling_point'  : 1,
        'vapor_pressure' : 2

    }

    for hidden_size in [512, 1024, 2048]:
        for num_layers in [1, 2, 3, 4]:
            for num_heads in [1, 2, 4, 8]:
                model_name = 'TCN_' + data_type
                model_name = f"{model_name}_{hidden_size}_{num_layers}_{num_heads}"
                
                trans_estimator = f"Hidden_size\t{hidden_size}\tNum_layer\t{num_layers}\tNum_heads\t{num_heads}"
                print(trans_estimator)
                
                df_copy = df.copy()
                ps = PredefinedSplit(test_fold=df_copy.cv_fold)
                param_dict['hidden_size'] = hidden_size
                param_dict['num_layers'] = num_layers
                param_dict['num_heads'] = num_heads
                param_dict['type'] = type_dict[data_type]
                param_dict['model_name'] = model_name 
                train_result(df_copy, ps, param_dict, device)

    
