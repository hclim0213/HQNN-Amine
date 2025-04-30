# File: /workspace/quantum_qspr/util/utility.py
import torch
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def create_param_dir_name(param_dict):
    """Create a directory name based on the parameter dictionary."""
    param_str = "_".join([f"{k}={v}" if not isinstance(v, list) else f"{k}={'-'.join(map(str, v))}" 
                          for k, v in param_dict.items()])
    return param_str

def numpy_to_python(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    else:
        return obj

def save_metrics_to_file(fold_metrics, param_dict, base_filename, save_dir):
    """Save metrics, parameter information, and data type to a JSON file."""
    metrics_data = {
        "data_type": base_filename,
        "parameters": param_dict,
        "fold_metrics": fold_metrics
    }
    
    metrics_data = numpy_to_python(metrics_data)
    
    param_str = "_".join(f"{k}={v}" for k, v in param_dict.items() if k != "hidden_sizes")
    hidden_sizes_str = "-".join(map(str, param_dict["hidden_sizes"]))
    filename = f"metrics_{base_filename}_{param_str}_hidden={hidden_sizes_str}.json"
    
    file_path = os.path.join(save_dir, filename)
    
    with open(file_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Metrics saved to {file_path}")

def load_and_preprocess_data(file_path, drop_columns, target_column, scale_target=True):
    """Load data from CSV or TSV and preprocess it."""
    # Check file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Load data based on file extension
    if file_extension == '.tsv':
        data = pd.read_csv(file_path, sep='\t')
    elif file_extension == '.csv':
        data = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Please use .csv or .tsv files.")
    
    # Separate features and target
    X = data.drop(drop_columns, axis=1)
    y = data[target_column]
    folds = data['cv_fold']
    
    # Scale target if required
    if scale_target:
        scaler = MinMaxScaler()
        y_scaled = pd.Series(scaler.fit_transform(y.values.reshape(-1, 1)).ravel(), name=f'{target_column}_scaled', index=y.index)
    else:
        y_scaled = y
        scaler = None
    
    return X, y, y_scaled, folds, scaler

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, save_path):
    """Save a PyTorch model."""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model_class, load_path, *args, **kwargs):
    """Load a PyTorch model."""
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    return model

def set_seeds(seed):
    """Set seeds for reproducibility."""
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# You can add more utility functions as needed