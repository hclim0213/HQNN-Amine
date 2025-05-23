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

class HQSc_DMPNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_qubits : int, 
        edge_size: int,
        steps: int,
        dropout: float,
        type : int, 
    ):
        super(HQSc_DMPNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_qubits = n_qubits 
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

class HQFx_DMPNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        edge_size: int,
        steps: int,
        dropout: float,
        type : int, 
        n_qubits : int = None, 
    ):
        super(HQFx_DMPNN, self).__init__()

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
