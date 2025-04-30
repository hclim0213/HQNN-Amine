import argparse
import logging
import torch
from sklearn.model_selection import PredefinedSplit

# Package imports
from mlp_utils.utility  import load_and_preprocess_data, ensure_dir
from mlp_utils.train_nn import perform_mlp
from mlp_utils.qnn     import Hybrid, CustomDataset     

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("hqnn_scratch")

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV/TSV with X + y + cv_fold")
    p.add_argument("--save-dir", default="hqnn_results")
    p.add_argument("--hidden-sizes", required=True,
                   help="Comma-sep list, e.g. 1024,512,256")
    p.add_argument("--n-qubits",  type=int, required=True)
    p.add_argument("--n-layers",  type=int, default=2)
    p.add_argument("--epochs",    type=int, default=300)
    p.add_argument("--device-type", default="lightning.qubit",
                   help="Passed straight into Hybrid.create_qnn()")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fold-col", default="cv_fold")
    p.add_argument("--target",   default="y")
    return p.parse_args()

def main() -> None:
    args = cli()
    ensure_dir(args.save_dir)

    X, y, y_scaled, folds, scaler = load_and_preprocess_data(
        args.data,
        drop_columns=[args.target, args.fold_col],
        target_column=args.target,
        scale_target=True,
    )
    ps = PredefinedSplit(folds)

    hidden_sizes = [int(h) for h in args.hidden_sizes.split(",")]

    param_grid = {
        "hidden_sizes" : [hidden_sizes],
        "batch_size"   : [128],
        "learning_rate": [1e-3],
        "num_epochs"   : [args.epochs],
        "dropout_rate" : [0.1],
        "n_qubit"      : [args.n_qubits],
        "n_layer"      : [args.n_layers],
        "device_type"  : [args.device_type],
    }

    best_params, _, best = perform_mlp(
        X, y, y_scaled, ps,
        param_grid   = param_grid,
        device       = args.device,
        save_dir     = args.save_dir,
        scaler       = scaler,
        model_class  = Hybrid,          # HQNN from scratch
        custom_dataset = CustomDataset,
    )
    LOG.info("Best R² product: %.4f", best)
    LOG.info("Best params    : %s", best_params)


if __name__ == "__main__":
    main()
