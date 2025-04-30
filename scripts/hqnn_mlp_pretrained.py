import argparse
import logging
import torch
from sklearn.model_selection import PredefinedSplit

from mlp_utils.utility  import load_and_preprocess_data, ensure_dir, load_model
from mlp_utils.train_nn import perform_mlp
from mlp_utils.qnn     import Hybrid, CustomDataset, Classical          # for typing

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("hqnn_pretrained")

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--pretrained-ckpt", required=True,
                   help=".pth file with a Classical backbone")
    p.add_argument("--save-dir", default="hqnn_pretrained_results")
    p.add_argument("--hidden-sizes", required=True)
    p.add_argument("--n-qubits",  type=int, required=True)
    p.add_argument("--n-layers",  type=int, default=2)
    p.add_argument("--epochs",    type=int, default=300)
    p.add_argument("--device-type", default="lightning.qubit")
    p.add_argument("--freeze-classical", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fold-col", default="cv_fold")
    p.add_argument("--target",   default="y")
    p.add_argument("--learning-rate",type=float,default=1e-3,help="Learning rate for finetuning")
    p.add_argument("--dropout-rate",type=float,default=0.1,help="Dropout rate for the classical layers")
    return p.parse_args()

def main():
    args = cli()
    ensure_dir(args.save_dir)

    # 1) data
    X, y, y_scaled, folds, scaler = load_and_preprocess_data(
        args.data,
        drop_columns=[args.target, args.fold_col],
        target_column=args.target,
        scale_target=True,
    )
    ps = PredefinedSplit(folds)

    # 2) load the pretrained classical backbone
    hidden = [int(h) for h in args.hidden_sizes.split(",")]
    backbone: Classical = load_model(
        model_class = Classical,
        load_path   = args.pretrained_ckpt,
        input_size  = X.shape[1],
        hidden_sizes= hidden,
        output_size = 1,
        dropout_rate= 0.1,
    )
    # replace head with n_qubits so Hybrid can consume it
    backbone.replace_output_layer(args.n_qubits)

    if args.freeze_classical:
        for p in backbone.parameters():
            p.requires_grad = False
        LOG.info("Classical backbone frozen.")

    pretrained_models = [backbone]   # same model reused across folds

    # 3) assemble a single-combo param grid
    param_grid = {
        "hidden_sizes":[hidden],
        "batch_size"  :[128],
        "learning_rate":[args.learning_rate],
        "num_epochs"  :[args.epochs],
        "dropout_rate":[args.dropout_rate],
        "n_qubit"     :[args.n_qubits],
        "n_layer"     :[args.n_layers],
        "device_type" :[args.device_type],
    }

    best_params, _, best = perform_mlp(
        X, y, y_scaled, ps,
        param_grid     = param_grid,
        device         = args.device,
        save_dir       = args.save_dir,
        scaler         = scaler,
        model_class    = Hybrid,
        custom_dataset = CustomDataset,
        pretrained_models = pretrained_models,
    )
    LOG.info("Best R² product = %.4f", best)
    LOG.info("Params = %s", best_params)


if __name__ == "__main__":
    main()
