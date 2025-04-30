import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pennylane as qml
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

class QNN(nn.Module):
    """Base class for classical and quantum neural networks."""

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        # self.model = self._build_model()

    def _build_model(self):
        """Builds a feedforward neural network."""
        layers = []
        prev_size = self.input_size
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, self.output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def arctan_encoding(x):
        """Apply arctan encoding transformation."""
        return torch.atan(x) + 0.5 * torch.pi

class Classical(QNN):
    """Fully classical regression model."""
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
        super().__init__(input_size, hidden_sizes, output_size, dropout_rate)
        self.model = self._build_model()

    def replace_output_layer(self, new_output_size):
        """Modify output layer to match a new output dimension."""
        in_features = self.model[-1].in_features
        self.model[-1] = nn.Linear(in_features, new_output_size)


class Hybrid(QNN):
    """Hybrid quantum-classical neural network."""

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate,
                 n_qubits, n_layers, device_type, pretrained_classical=None, token_str=None):
        # Use n_qubits as the output dimension for the classical part.
        super().__init__(input_size, hidden_sizes, n_qubits, dropout_rate)
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_type = device_type
        self.token_str = token_str
        # Pass the necessary parameters to create_qnn.
        self.q_layer = self.create_qnn(self.n_qubits, self.n_layers, self.device_type, self.token_str)
        torch.manual_seed(42)  
        with torch.no_grad():
            self.q_layer.weights.data.uniform_(-np.pi, np.pi)
        # print(f"🔍 Initial QLayer weights: {self.q_layer.weights}")

        # Use a pretrained classical model if available.
        if pretrained_classical:
            self.classical_layers = pretrained_classical
            # print("🔍 Loaded Pretrained Classical Weights into Hybrid:")
            # for name, param in self.classical_layers.named_parameters():
            #     print(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        else:
            self.classical_layers = self._build_model()

    @classmethod
    def create_qnn(cls, n_qubits, n_layers, device_type="lightning.qubit", token_str=None):
        """Create a quantum neural network (QNN) circuit using PennyLane."""
        #noise parameters on 2025.Jan.10th, except Fez (2024.Oct.29th)
        device_noise_parameters = {
            "ibm_yonsei": {
                "two_qubit_error": 3.89e-2, "median_sx_error": 2.944e-4, "readout_error": 2.080e-2, "t1": 2.4152e-4, "t2": 1.5397e-4, "delta_t": 8.4e-8
            },
            "ibm_brussels": {
                "two_qubit_error": 2.86e-2, "median_sx_error": 2.822e-4, "readout_error": 2.420e-2, "t1": 2.6671e-4, "t2": 1.2221e-4, "delta_t": 6.6e-7
            },
            "ibm_strasbourg": {
                "two_qubit_error": 2.91e-2, "median_sx_error": 2.649e-4, "readout_error": 1.840e-2, "t1": 2.6645e-4, "t2": 1.4879e-4, "delta_t": 6.6e-7
            },
            "ibm_torino": {
                "two_qubit_error": 6.25e-3, "median_sx_error": 3.508e-4, "readout_error": 2.000e-2, "t1": 1.6605e-4, "t2": 1.3578e-4, "delta_t": 6.8e-8
            },
            "ibm_marrakesh": {
                "two_qubit_error": 3.41e-3, "median_sx_error": 2.460e-4, "readout_error": 1.540e-2, "t1": 1.7998e-4, "t2": 1.1391e-4, "delta_t": 6.8e-8
            },
            "ibm_fez": {
                "two_qubit_error": 2.792e-3, "median_sx_error": 2.703e-4, "readout_error": 1.645e-2, "t1": 1.1806e-4, "t2": 9.141e-5, "delta_t": 6.8e-8
            },
            "ibm_brisbane": {
                "two_qubit_error": 1.65e-2, "median_sx_error": 2.549e-4, "readout_error": 1.440e-2, "t1": 2.2387e-4, "t2": 1.3951e-4, "delta_t": 6.6e-7
            }
        }
        if device_type in device_noise_parameters:
            params = device_noise_parameters[device_type]
            two_qubit_depolarizing_error = params["two_qubit_error"]
            single_qubit_depolarizing_error = params["median_sx_error"]
            amplitude_damping_prob = 1 - np.exp(-params["delta_t"] / params["t1"])
            phase_damping_prob = 1 - np.exp(-params["delta_t"] / params["t2"])
            metadata = dict(two_qubit_depolarizing_error=two_qubit_depolarizing_error, 
                            single_qubit_depolarizing_error=single_qubit_depolarizing_error, 
                             amplitude_damping_prob=amplitude_damping_prob, 
                            phase_damping_prob=phase_damping_prob)
                
            def noise_ry(op, **kwargs):
                for wire in op.wires:
                    qml.DepolarizingChannel(single_qubit_depolarizing_error, wires=wire)
                    qml.AmplitudeDamping(amplitude_damping_prob, wires=wire)
                    qml.PhaseDamping(phase_damping_prob, wires=wire)
            
            def noise_cnot(op, **kwargs):
                for wire in op.wires:
                    qml.DepolarizingChannel(two_qubit_depolarizing_error, wires=wire)
                    qml.AmplitudeDamping(amplitude_damping_prob, wires=wire)
                    qml.PhaseDamping(phase_damping_prob, wires=wire)
            
            fcond = qml.noise.op_eq("RY") & qml.noise.wires_in(list(range(n_qubits)))
            fcond2 = qml.noise.op_eq("CNOT") & qml.noise.wires_in(list(range(n_qubits)))
            noise_model = qml.NoiseModel({fcond: noise_ry, fcond2: noise_cnot}, **metadata)
            dev = qml.device('default.mixed', wires=n_qubits, readout_prob=params["readout_error"])
            
        elif device_type == "qiskit.remote":
            if token_str ==None:
                print('Please provide your IBM quantum token')
            QiskitRuntimeService.save_account(channel='ibm_quantum', token=token_str, overwrite=True)
            service = QiskitRuntimeService()
            backend = service.backend(name='ibm_yonsei')        
            dev = qml.device("qiskit.remote", wires=127, backend=backend, seed_transpiler=42)
            noise_model = None
        else:
            dev = qml.device(device_type, wires=n_qubits)
            noise_model = None

        @qml.qnode(dev)
        def circuit(inputs, weights):
            encoded_inputs = cls.arctan_encoding(inputs)
            for i in range(n_qubits):
                qml.RY(encoded_inputs[i], wires=i)
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(weights[layer * n_qubits + i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        if noise_model:
            # print('noise added')
            circuit=qml.add_noise(circuit, noise_model)
            print('noise added')


        return qml.qnn.TorchLayer(qnode=circuit, weight_shapes={"weights": n_layers * n_qubits})

    def forward(self, x):
        classical_output = self.classical_layers(x)
        quantum_output = torch.stack([self.q_layer(inp).sum() for inp in classical_output])
        return quantum_output

class CustomDataset(Dataset):
    """Custom Dataset for loading features and targets."""
    def __init__(self, X, y, y_scaled):
        self.X = X
        self.y = y
        self.y_scaled = y_scaled

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.y_scaled.iloc[idx], dtype=torch.float32)
        original_target = torch.tensor(self.y.iloc[idx], dtype=torch.float32)
        return data, target, original_target
