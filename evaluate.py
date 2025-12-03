import argparse
from pathlib import Path

import numpy as np
import torch

import config
import data_loading
import metrics
import models


def maybe_load(model, ckpt_path, device):
    if ckpt_path is None:
        return None
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"Checkpoint {ckpt} not found; skipping load.")
        return None
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Streaming evaluation across many scenes.")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT))
    parser.add_argument("--prefer-lazy", action="store_true", help="Use lazy zarr arrays (default).")
    parser.add_argument("--load-into-ram", action="store_true", help="Force loading tables into RAM.")
    parser.add_argument("--max-samples", type=int, default=100000, help="Total samples to stream through.")
    parser.add_argument("--samples-per-scene", type=int, default=30, help="Max samples drawn from each scene.")
    parser.add_argument("--scene-count", type=int, default=None, help="Randomly choose N scenes (default: all).")
    parser.add_argument("--history", type=int, default=config.SAFE_H)
    parser.add_argument("--horizon", type=int, default=config.SAFE_T)
    parser.add_argument("--kmax", type=int, default=config.SAFE_KMAX)
    parser.add_argument("--hidden", type=int, default=config.SAFE_HIDDEN)
    parser.add_argument("--k-modes", type=int, default=config.K_MODES)
    parser.add_argument("--cnn-ckpt", type=str, default="artifacts/cnn.pt")
    parser.add_argument("--gnn-ckpt", type=str, default="artifacts/gnn.pt")
    parser.add_argument("--gnn-multi-ckpt", type=str, default="artifacts/gnn_multi.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # Keep config constants aligned with the checkpoint settings.
    config.SAFE_H = args.history
    config.SAFE_T = args.horizon
    config.SAFE_KMAX = args.kmax
    config.K_MODES = args.k_modes

    config.set_seeds(args.seed)
    device = args.device or config.get_device()
    print("Using device:", device)

    prefer_lazy = True
    if args.load_into_ram:
        prefer_lazy = False
    elif args.prefer_lazy:
        prefer_lazy = True

    tables = data_loading.open_tables(
        Path(args.data_root),
        prefer_lazy=prefer_lazy,
    )

    cnn = models.CNN1D(T_out=args.horizon)
    cnn = maybe_load(cnn, args.cnn_ckpt, device)

    gnn = None
    if args.gnn_ckpt:
        gnn = models.GNNMotion(h=args.hidden, T=args.horizon)
        gnn = maybe_load(gnn, args.gnn_ckpt, device)

    gnn_multi = None
    if args.gnn_multi_ckpt:
        gnn_multi = models.GNNMultiMotion(h=args.hidden, T=args.horizon, K=args.k_modes)
        gnn_multi = maybe_load(gnn_multi, args.gnn_multi_ckpt, device)

    if cnn is None or gnn is None or gnn_multi is None:
        raise SystemExit("Ensure CNN, GNN, and GNN-Multi checkpoints are available before streaming eval.")

    scene_indices = None
    if args.scene_count is not None:
        rng = np.random.default_rng(args.seed)
        all_idx = np.arange(len(tables.scenes))
        rng.shuffle(all_idx)
        scene_indices = all_idx[: args.scene_count]

    results = metrics.streaming_eval_over_scenes(
        tables=tables,
        cnn_model=cnn,
        gnn_model=gnn,
        gnn_multi_model=gnn_multi,
        max_global_samples=args.max_samples,
        scenes_to_visit=scene_indices,
        max_samples_per_scene=args.samples_per_scene,
        device=device,
    )

    print("\n=== Streaming evaluation ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
