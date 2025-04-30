from typing import Dict

import torch
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add

class DirectedMessagePassingLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        edge_in: int,
        steps: int,
        dropout: float,
    ):
        super(DirectedMessagePassingLayer, self).__init__()
        self.norm_layer = torch.nn.LayerNorm(output_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()

        self.dense_init = torch.nn.Linear(input_size + edge_in, output_size)
        self.dense_hidden = torch.nn.Sequential(
            torch.nn.Linear(output_size, output_size),
            torch.nn.ReLU(),
            torch.nn.Linear(output_size, output_size),
        )
        self.dense_final = torch.nn.Linear(input_size + output_size, output_size)
        self.steps = steps

    def forward(
        self,
        x: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.apply_conv(x, edge_feats, edge_index)
        x = self.norm_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def apply_conv(self, x: torch.Tensor, edge_feats, edge_index):
        edge_index = edge_index
        edge_attr = edge_feats.float()

        h0 = torch.cat((x[edge_index[0, :]], edge_attr), dim=1)
        h0 = self.activation(self.dense_init(h0))

        h = h0
        for step in range(self.steps):
            h_ = scatter_add(h, edge_index[1, :], dim=0)
            m = h_[edge_index[1, :]] - h
            h = self.activation(h0 + self.dense_hidden(m))

        m = scatter_add(h, edge_index[0, :], dim=0, dim_size=x.size(0))
        h = self.activation(self.dense_final(torch.cat((x, m), dim=1)))
        return h

class DirectedMessagePassingNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        edge_size: int,
        steps: int,
        dropout: float,
        type : int, 
    ):
        super(DirectedMessagePassingNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.edge_size = edge_size
        self.steps = steps
        self.dropout = dropout

        activation = torch.nn.ReLU

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(
            DirectedMessagePassingLayer(
                input_size, hidden_size, edge_size, steps, dropout
            )
        )

        self.type = type
        
        if type == 1:
            hidden_size = hidden_size    
        elif type == 2:
            hidden_size = hidden_size + 1

        self.final_dense = torch.nn.Linear(hidden_size, output_size, bias=False)

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
            x = layer(node_feats, edge_feats, edge_index)

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
            edge_size=self.edge_size,
            steps=self.steps,
            dropout=self.dropout,
        )

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
        

class EGConvNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        type : int, 
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

        self.type = type
        
        if type == 1:
            hidden_size = hidden_size    
        elif type == 2:
            hidden_size = hidden_size + 1

        self.final_dense = torch.nn.Linear(hidden_size, output_size, bias=False)

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

from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch_geometric.nn import GraphConv, global_mean_pool

class GraphConvLayerV2(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = GraphConv(in_features, out_features)

        self.res_connection = torch.nn.Linear(in_features, out_features)
        self.norm_layer = torch.nn.LayerNorm(out_features)

        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x, edge_index)

        # residual connection is applied before normalization and activation
        # according to https://arxiv.org/pdf/2006.07739.pdf
        x += self.res_connection(identity)
        x = self.norm_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
        
class GraphConvNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        type : int, 
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        activation = torch.nn.ReLU

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(GraphConvLayerV2(input_size, hidden_size, dropout=dropout))
        for i in range(num_layers - 1):
            self.gnn_layers.append(
                GraphConvLayerV2(hidden_size, hidden_size, dropout=dropout)
            )


        self.type = type

        if type == 1:
            hidden_size = hidden_size    
        elif type == 2:
            hidden_size = hidden_size + 1
            
        self.final_dense = torch.nn.Linear(hidden_size, output_size, bias=False)

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
            dropout=self.dropout,
        )


from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch_geometric.nn import TransformerConv, global_mean_pool

class TransformerConvLayer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_heads: int = 1,
        alpha: float = 0.2,
        concat: bool = True,
        dropout: float = 0.0,
    ):
        super(TransformerConvLayer, self).__init__()
        self.conv = TransformerConv(input_size, output_size // num_heads, heads=num_heads, concat=concat, dropout=dropout)
        
        self.res_connection = torch.nn.Linear(input_size, output_size)
        self.norm_layer = torch.nn.LayerNorm(output_size)

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

class TransformerConvNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        type : int, 
    ):
        super(TransformerConvNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        activation = torch.nn.ReLU

        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_layers.append(TransformerConvLayer(input_size, hidden_size, num_heads, dropout=dropout))
        for i in range(num_layers - 1):
            self.gnn_layers.append(
                TransformerConvLayer(hidden_size, hidden_size, num_heads, dropout=dropout)
            )

        self.type = type

        if type == 1:
            hidden_size = hidden_size    
        elif type == 2:
            hidden_size = hidden_size + 1
            
        self.final_dense = torch.nn.Linear(hidden_size, output_size, bias=False)

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

from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATV2Layer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        alpha: float = 0.2,
        concat: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = GATv2Conv(in_features, out_features // num_heads, heads=num_heads)
        
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
        
class GraphAttnTransformer(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        type: int
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
        self.gnn_layers.append(GATV2Layer(input_size, hidden_size, num_heads, dropout=dropout))
        for i in range(num_layers - 1):
            self.gnn_layers.append(
                GATV2Layer(hidden_size, hidden_size, num_heads, dropout=dropout)
            )

        self.type = type 
        
        if type == 1:
            hidden_size = hidden_size    
        elif type == 2:
            hidden_size = hidden_size + 1
        self.final_dense = torch.nn.Linear(hidden_size, output_size, bias=False)

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