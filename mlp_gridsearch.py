from pathlib import Path
import argparse
import logging
import torch
from sklearn.model_selection import PredefinedSplit

from mlp_utils.utility  import load_and_preprocess_data, ensure_dir
from mlp_utils.train_nn import perform_mlp

LOG = logging.getLogger("mlp_gs")
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s: %(message)s")

PARAM_GRID = {                       # 24 combos = 6×2×2
    "hidden_sizes": [
        [2048, 1024, 512, 256, 128],
        [2048, 1024, 512, 256],
        [2048, 1024, 512],
        [1024, 1024, 1024],
        [1024, 512, 256, 128],
        [1024, 512, 256],
    ],
    "batch_size"   : [128],         
    "learning_rate": [0.01, 0.001],
    "num_epochs"   : [300],          
    "dropout_rate" : [0.1, 0.2],
}

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="CSV/TSV with features + y + cv_fold")
    p.add_argument("--save-dir", default="grid_results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target", default="y")
    p.add_argument("--fold-col", default="cv_fold")
    return p.parse_args()

def main() -> None:
    args = cli()
    ensure_dir(args.save_dir)

    X, y, y_scaled, folds, scaler = load_and_preprocess_data(
        args.data,
        drop_columns=[args.target, args.fold_col],
        target_column=args.target,
        scale_target=True
    )
    ps = PredefinedSplit(folds)

    best_params, all_results, best = perform_mlp(
        X, y, y_scaled, ps,
        param_grid = PARAM_GRID,
        device     = args.device,
        save_dir   = args.save_dir,
        scaler     = scaler,
        base_filename = Path(args.data).stem,
    )

    LOG.info("Best R² product: %.4f", best)
    LOG.info("Best params   : %s", best_params)

if __name__ == "__main__":
    main()