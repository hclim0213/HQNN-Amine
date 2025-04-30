from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch_geometric.nn import EGConv, global_mean_pool

class EGConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        alpha: float = 0.2,
        concat: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = EGConv(in_features, out_features, num_heads=num_heads)
        
        self.res_connection = torch.nn.Linear(in_features, out_features)
        self.norm_layer = torch.nn.LayerNorm(out_features)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x, edge_index)
        
        x += self.res_connection(identity)
        x = self.norm_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
        
class HQSc_EGC(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_qubits: int, 
        num_layers: int,
        num_heads: int,
        dropout: float,
        type : int, 
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_qubits = n_qubits 
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        activation = torch.nn.ReLU

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(EGConvLayer(input_size, hidden_size, num_heads, dropout=dropout))
        for i in range(num_layers - 1):
            self.gnn_layers.append(
                EGConvLayer(hidden_size, hidden_size, num_heads, dropout=dropout)
            )

        self.type = type
        embed_dim = hidden_size if type == 1 else hidden_size + 1
            
        self.final_dense = torch.nn.Linear(embed_dim, n_qubits, bias=False)

    def forward(  # type: ignore
        self,
        data : torch.Tensor, 
        return_activations: bool = False,
    ) -> torch.Tensor:

        node_feats = data.x
        edge_feats = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch 

        x = node_feats.float()
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)

        if return_activations:
            return x

        x = global_mean_pool(x, batch)

        if self.type == 2:
            x = torch.concat((x, data.temp.unsqueeze(1)), dim=1)
        x = self.final_dense(x)

        return x

    def config(self) -> Dict:
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

class HQFx_EGC(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        type : int, 
        n_qubits : int = None, 
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        activation = torch.nn.ReLU

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(EGConvLayer(input_size, hidden_size, num_heads, dropout=dropout))
        for i in range(num_layers - 1):
            self.gnn_layers.append(
                EGConvLayer(hidden_size, hidden_size, num_heads, dropout=dropout)
            )

        embed_dim = hidden_size if type == 1 else hidden_size + 1
        self.type = type
            
        if n_qubits != None:
            self.final_dense = torch.nn.Sequential(torch.nn.Linear(embed_dim, n_qubits, bias=False),
                                                  torch.nn.Linear(n_qubits, output_size, bias=False)
                                                  )
        elif n_qubits == None:
            self.final_dense = torch.nn.Linear(embed_dim, output_size, bias=False)

    def forward(  # type: ignore
        self,
        data : torch.Tensor, 
        return_activations: bool = False,
    ) -> torch.Tensor:

        node_feats = data.x
        edge_feats = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch 

        x = node_feats.float()
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)

        if return_activations:
            return x

        x = global_mean_pool(x, batch)

        if self.type == 2:
            x = torch.concat((x, data.temp.unsqueeze(1)), dim=1)
        x = self.final_dense(x)

        return x

    def config(self) -> Dict:
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )