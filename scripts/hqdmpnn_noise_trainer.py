import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric import loader
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import MinMaxScaler

from src.training.HQGNN_commons import smiles2tensor, train, load_model, get_device_metadata
from src.models.HQDMPNN import HQSc_DMPNN, HQFx_DMPNN
from src.models.HQGNN import QML_hybrid_graph, QML_hybrid_pretrained_graph

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_param_dict(args, data_type):
    type_dict = {
        'pKa': 1, 'viscosity': 2, 'melting_point': 1,
        'boiling_point': 1, 'vapor_pressure': 2
    }

    param_dict = {
        'input_size': 84,
        'hidden_size': 512,
        'steps': 1,
        'output_size': 1,
        'dropout': 0.1,
        'lr': 1e-3,
        'batch_size': 128,
        'epochs': args.epochs,
        'n_qubits': args.n_qubits,
        'n_layers': args.n_layers,
        'model_name': args.model_name,
        'q_device': args.q_device,
        'type': type_dict[data_type],
        'edge_size': 12,
        'mode': args.mode,
    }

    if args.mode == 'transfer':
        param_dict.update({
            'pretrained_model_path': args.pretrained_model_path,
            'pretrained_model_name': os.path.splitext(os.path.basename(args.pretrained_model_path))[0],
            'requires_grad': args.requires_grad
        })

    return param_dict

def prepare_data(df):
    scaler = MinMaxScaler((0, 1))
    df['y_scaled'] = scaler.fit_transform(df[['y']])
    ps = PredefinedSplit(test_fold=df.cv_fold)
    return df, ps, scaler

def build_model(cv, param_dict, device, metadata):
    if param_dict['mode'] == 'scratch':
        graph = HQSc_DMPNN(
            input_size=param_dict['input_size'],
            output_size=param_dict['output_size'],
            hidden_size=param_dict['hidden_size'],
            edge_size=param_dict['edge_size'],
            steps=param_dict['steps'],
            n_qubits=param_dict['n_qubits'],
            dropout=param_dict['dropout'],
            type=param_dict['type']
        ).to(device)

        return QML_hybrid_graph(
            input_size=param_dict['input_size'],
            hidden_sizes=param_dict['hidden_size'],
            output_size=param_dict['output_size'],
            dropout=param_dict['dropout'],
            n_qubits=param_dict['n_qubits'],
            n_layers=param_dict['n_layers'],
            graph=graph,
            q_device=param_dict['q_device'],
            metadata=metadata
        ).to(device)

    elif param_dict['mode'] == 'transfer':
        pretrained = HQFx_DMPNN(
            input_size=param_dict['input_size'],
            output_size=param_dict['output_size'],
            hidden_size=param_dict['hidden_size'],
            edge_size=param_dict['edge_size'],
            steps=param_dict['steps'],
            dropout=param_dict['dropout'],
            type=param_dict['type']
        ).to(device)

        pretrained_weight = load_model(file_path=param_dict['pretrained_model_path'], model_name=f"{param_dict['pretrained_model_name']}_{cv}", device=device)
        pretrained.load_state_dict(pretrained_weight)
        pretrained.final_dense = nn.Identity()

        for p in pretrained.parameters():
            p.requires_grad = param_dict['requires_grad']

        return QML_hybrid_pretrained_graph(
            input_size=param_dict['input_size'],
            hidden_size=param_dict['hidden_size'],
            output_size=param_dict['output_size'],
            dropout=param_dict['dropout'],
            n_qubits=param_dict['n_qubits'],
            n_layers=param_dict['n_layers'],
            graph=pretrained,
            type=param_dict['type'],
            q_device=param_dict['q_device'],
            metadata=metadata
        ).to(device)

def train_result(df, ps, param_dict, device, scaler, metadata):
    for cv, (train_idx, valid_idx) in enumerate(ps.split()):
        train_data = smiles2tensor(df.iloc[train_idx], param_dict['type'])
        valid_data = smiles2tensor(df.iloc[valid_idx], param_dict['type'])

        train_dataloader = loader.DataLoader(train_data, batch_size=param_dict['batch_size'], shuffle=True)
        valid_dataloader = loader.DataLoader(valid_data, batch_size=param_dict['batch_size'], shuffle=True)

        model = build_model(cv, param_dict, device, metadata)
        optimizer = optim.Adam(model.parameters(), lr=param_dict['lr'])
        loss_fn = nn.MSELoss()

        best_train_metrics, best_val_metrics, best_epoch = train(
            model, train_dataloader, valid_dataloader, optimizer, loss_fn, 
            param_dict['epochs'], device, param_dict['model_name'], cv, scaler
        )

        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tTrain\tR2\t{best_train_metrics['r2']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tTrain\tRMSE\t{best_train_metrics['rmse']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tTrain\tMAE\t{best_train_metrics['mae']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tTrain\tMSE\t{best_train_metrics['mse']}")
        
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tValid\tR2\t{best_val_metrics['r2']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tValid\tRMSE\t{best_val_metrics['rmse']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tValid\tMAE\t{best_val_metrics['mae']}")
        print(f"CV\t{cv}\tEpoch\tBest_Train_Scores\t{best_epoch}\tValid\tMSE\t{best_val_metrics['mse']}")

def main():
    seed_everything()
    parser = argparse.ArgumentParser(description="HQDMPNN Training")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Shared args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data_file', required=True)
    common.add_argument('--n_qubits', type=int, default=4)
    common.add_argument('--n_layers', type=int, default=3)
    common.add_argument('--epochs', type=int, default=300)
    common.add_argument('--model_name', type=str, required=True)
    common.add_argument('--q_device', type=str, default='default.mixed')
    common.add_argument('--hw_device', type=str, default='fez', help='hardware noise')


    scratch = subparsers.add_parser('scratch', parents=[common])
    transfer = subparsers.add_parser('transfer', parents=[common])
    transfer.add_argument('--pretrained_model_path', required=True)
    transfer.add_argument('--requires_grad', type=bool, default=True)

    args = parser.parse_args()
    df = pd.read_csv(args.data_file, sep='\t')
    data_type = os.path.basename(args.data_file).split('.')[0]

    param_dict = setup_param_dict(args, data_type)
    df_copy, ps, scaler = prepare_data(df.copy())
    device = torch.device('cpu')  # or use 'cuda' if needed
    metadata = get_device_metadata(args.hw_device)

    start = datetime.now()
    train_result(df_copy, ps, param_dict, device, scaler, metadata)
    print(f"Total time: {datetime.now() - start}")

if __name__ == "__main__":
    main()
