import numpy as np
import random 
import torch
import torch.nn as nn
import pennylane as qml

import warnings
warnings.filterwarnings('ignore')

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class RYAngleLayer(nn.Module):
    def forward(self, x):
        return torch.atan(x) + 0.5 * torch.pi
        
def create_qnn(n_qubits, n_layers, q_device):
    # dev = qml.device("default.qubit", wires=n_qubits)
    dev = qml.device(q_device, wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(inputs, weights):
        encoded_inputs = RYAngleLayer()(inputs)
        # Feature map
        for i in range(n_qubits):
            qml.RY(encoded_inputs[i], wires=i)
        
        # Ansatz
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer * n_qubits + i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    weight_shapes = {"weights": n_layers * n_qubits} 
    
    return circuit, weight_shapes

""" noise part """
def noise_ry(op, **kwargs):
    metadata = kwargs.get("metadata", {})
    for wire in op.wires:
        qml.DepolarizingChannel(metadata["single_qubit_depolarizing_error"], wires=wire)
        qml.AmplitudeDamping(metadata["amplitude_damping_prob"], wires=wire)
        qml.PhaseDamping(metadata["phase_damping_prob"], wires=wire)

def noise_cnot(op, **kwargs):
    metadata = kwargs.get("metadata", {})
    for wire in op.wires:
        qml.DepolarizingChannel(metadata["two_qubit_depolarizing_error"], wires=wire)
        qml.AmplitudeDamping(metadata["amplitude_damping_prob"], wires=wire)
        qml.PhaseDamping(metadata["phase_damping_prob"], wires=wire)

def noise_qnn(n_qubits, n_layers, q_device, metadata):
    dev = qml.device(q_device, wires=n_qubits, readout_prob=metadata['readout_prob'])
    
    # nosie part 
    fcond = qml.noise.op_eq("RY") & qml.noise.wires_in(list(range(n_qubits)))
    fcond2 = qml.noise.op_eq("CNOT") & qml.noise.wires_in(list(range(n_qubits)))
    noise_model = qml.NoiseModel({fcond: noise_ry, fcond2: noise_cnot}, metadata=metadata)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        encoded_inputs = RYAngleLayer()(inputs)
        # Feature map
        for i in range(n_qubits):
            qml.RY(encoded_inputs[i], wires=i)
        
        # Ansatz
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer * n_qubits + i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    ideal_circuit = qml.QNode(circuit, dev)
    noisy_circuit = qml.add_noise(ideal_circuit, noise_model)
    
    weight_shapes = {"weights": n_layers * n_qubits} # random 값 조절     
    return noisy_circuit, weight_shapes


class QML_hybrid_pretrained_graph(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 dropout, 
                 n_qubits, 
                 n_layers, 
                 graph, 
                 q_device,
                 type, 
                 metadata=None):
    
        super(QML_hybrid_pretrained_graph, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.graph = graph

        # Quantum layer
        if metadata == None:
            circuit, weight_shapes = create_qnn(n_qubits, n_layers, q_device)
        elif metadata != None:
            circuit, weight_shapes = noise_qnn(n_qubits, n_layers, q_device, metadata)

        if type == 2:
            self.pre_q = nn.Linear(hidden_size + 1, n_qubits)
        elif type == 1:
            self.pre_q = nn.Linear(hidden_size, n_qubits)
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, data):
        graph_output = self.graph(data) # final size = 4 or 9 
        graph_output = self.pre_q(graph_output)
        quantum_output = torch.stack([self.q_layer(i).sum() for i in graph_output])
        return quantum_output 

class QML_hybrid_graph(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_sizes, 
                output_size, 
                dropout, 
                n_qubits, 
                n_layers, 
                graph, 
                q_device,
                metadata=None):
    
        super(QML_hybrid_graph, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.graph = graph

        # Quantum layer
        if metadata == None:
            circuit, weight_shapes = create_qnn(n_qubits, n_layers, q_device)
        elif metadata != None:
            circuit, weight_shapes = noise_qnn(n_qubits, n_layers, q_device, metadata)

        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, data):
        graph_output = self.graph(data)
        quantum_output = torch.stack([self.q_layer(i).sum() for i in graph_output])
        return quantum_output 