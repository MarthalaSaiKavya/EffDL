import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config
import data_loading
import datasets
import metrics
import models
import sampling


def split_samples(items: List[dict], seed: int, train_ratio: float = 0.8) -> Tuple[List[dict], List[dict]]:
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    cut = int(train_ratio * len(idx))
    train_s = [items[i] for i in idx[:cut]]
    val_s = [items[i] for i in idx[cut:]]
    return train_s, val_s


def iter_batches(items: List[dict], bs: int, seed: int):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i : i + bs]]


def train_cnn_model(train_s, val_s, device: str, epochs: int, batch_size: int, horizon: int):
    config.clear_mem()
    dl_tr, dl_va = datasets.make_target_only_loaders(train_s, val_s, batch_size=batch_size)
    cnn = models.CNN1D(T_out=horizon).to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=3e-4)

    for ep in range(1, epochs + 1):
        cnn.train()
        run_loss = 0.0
        n = 0
        for x, y in dl_tr:
            config.clear_mem()
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            p = cnn(x)
            loss = F.smooth_l1_loss(p, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)

        cnn.eval()
        with torch.no_grad():
            vloss = 0.0
            vn = 0
            all_ade = 0.0
            all_fde = 0.0
            for x, y in dl_va:
                x = x.to(device)
                y = y.to(device)
                p = cnn(x)
                vloss += F.smooth_l1_loss(p, y).item() * x.size(0)
                ade, fde = metrics.ade_fde_batch(p, y)
                all_ade += ade * x.size(0)
                all_fde += fde * x.size(0)
                vn += x.size(0)
        print(
            f"[CNN {ep}/{epochs}] train={run_loss/max(1,n):.3f} | "
            f"val={vloss/max(1,vn):.3f} | ADE={all_ade/max(1,vn):.2f} | FDE={all_fde/max(1,vn):.2f}"
        )
    return cnn


def train_gnn_model(train_s, val_s, device: str, epochs: int, batch_size: int, hidden: int, horizon: int, seed: int):
    config.clear_mem()
    gnn = models.GNNMotion(h=hidden, T=horizon).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=3e-4)

    for ep in range(1, epochs + 1):
        gnn.train()
        losses = []
        for batch in iter_batches(train_s, batch_size, seed + ep):
            config.clear_mem()
            opt.zero_grad()
            total = 0.0
            for s in batch:
                X = torch.tensor(s["X"], dtype=torch.float32).to(device)
                EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
                EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
                y = torch.tensor(s["x_fut"], dtype=torch.float32).to(device)
                p = gnn(X, EI, EA, s["target_index"])
                total = total + F.smooth_l1_loss(p, y)
            total = total / max(1, len(batch))
            total.backward()
            opt.step()
            losses.append(total.item())

        gnn.eval()
        with torch.no_grad():
            ade, fde = metrics.eval_gnn_single(gnn, val_s, device=device, max_cases=500)
        print(f"[GNN {ep}/{epochs}] loss={np.mean(losses):.3f} | ADE={ade:.2f} | FDE={fde:.2f}")
    return gnn


def train_gnn_multi_model(
    train_s, val_s, device: str, epochs: int, batch_size: int, hidden: int, horizon: int, seed: int, k_modes: int
):
    config.clear_mem()
    gnn_multi = models.GNNMultiMotion(h=hidden, T=horizon, K=k_modes).to(device)
    opt = torch.optim.Adam(gnn_multi.parameters(), lr=3e-4)

    for ep in range(1, epochs + 1):
        gnn_multi.train()
        train_losses = []
        for batch in iter_batches(train_s, batch_size, seed + ep):
            config.clear_mem()
            opt.zero_grad()

            ys = []
            mus = []
            log_vars = []
            logits = []
            for s in batch:
                X = torch.tensor(s["X"], dtype=torch.float32).to(device)
                EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
                EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
                y = torch.tensor(s["x_fut"], dtype=torch.float32).to(device)

                mu, log_var, logit = gnn_multi(X, EI, EA, s["target_index"])
                ys.append(y.unsqueeze(0))
                mus.append(mu)
                log_vars.append(log_var)
                logits.append(logit)

            Y = torch.cat(ys, dim=0)
            MU = torch.cat(mus, dim=0)
            LV = torch.cat(log_vars, dim=0)
            LOG = torch.cat(logits, dim=0)

            loss = metrics.mixture_nll(Y, MU, LV, LOG)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        gnn_multi.eval()
        with torch.no_grad():
            val_losses = []
            for s in val_s[:500]:
                X = torch.tensor(s["X"], dtype=torch.float32).to(device)
                EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
                EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
                y = torch.tensor(s["x_fut"], dtype=torch.float32).unsqueeze(0).to(device)
                mu, log_var, logits = gnn_multi(X, EI, EA, s["target_index"])
                val_losses.append(metrics.mixture_nll(y, mu, log_var, logits).item())
        print(f"[GNN-Multi {ep}/{epochs}] train NLL={np.mean(train_losses):.3f} | val NLL={np.mean(val_losses):.3f}")
    return gnn_multi


def main():
    parser = argparse.ArgumentParser(description="Lyft motion forecasting (CNN + GNN) pipeline")
    parser.add_argument("--data-root", type=str, default=str(config.DATA_ROOT), help="Root containing scenes/*.zarr")
    parser.add_argument("--prefer-lazy", action="store_true", help="Use lazy zarr arrays (default).")
    parser.add_argument("--load-into-ram", action="store_true", help="Force loading tables into RAM.")
    parser.add_argument("--load-agents-into-ram", action="store_true", help="Load agents table fully (careful: large).")
    parser.add_argument("--max-samples", type=int, default=config.SAFE_MAX_SAMPLES, help="Total samples to draw.")
    parser.add_argument("--scenes", type=int, default=None, help="How many scenes to pull from (default=auto).")
    parser.add_argument("--frames-per-scene", type=int, default=config.SAFE_FRAMES_PER_SC, help="Frame stride per scene.")
    parser.add_argument("--history", type=int, default=config.SAFE_H, help="History length H.")
    parser.add_argument("--horizon", type=int, default=config.SAFE_T, help="Prediction horizon T.")
    parser.add_argument("--kmax", type=int, default=config.SAFE_KMAX, help="Neighbor cap.")
    parser.add_argument("--radius", type=float, default=30.0, help="Neighbor search radius (meters).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--cnn-epochs", type=int, default=3)
    parser.add_argument("--gnn-epochs", type=int, default=3)
    parser.add_argument("--gnn-multi-epochs", type=int, default=3)
    parser.add_argument("--cnn-batch", type=int, default=config.SAFE_BATCH_CNN)
    parser.add_argument("--gnn-batch", type=int, default=32)
    parser.add_argument("--gnn-multi-batch", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=config.SAFE_HIDDEN, help="Hidden width for GNNs.")
    parser.add_argument("--k-modes", type=int, default=config.K_MODES, help="Number of futures for multi-modal head.")
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto).")
    parser.add_argument("--save-dir", type=str, default="artifacts", help="Directory to save checkpoints.")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip training the single-mode GNN.")
    parser.add_argument("--skip-gnn-multi", action="store_true", help="Skip training the multi-modal GNN.")
    args = parser.parse_args()

    # Keep shared config constants aligned with the chosen hyperparameters.
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
        load_agents_into_ram=args.load_agents_into_ram,
    )
    print("Using Zarr split at", tables.zarr_root)

    samples = sampling.collect_samples(
        tables,
        max_samples=args.max_samples,
        scenes_to_use=args.scenes,
        frames_per_scene=args.frames_per_scene,
        H=args.history,
        T=args.horizon,
        R=args.radius,
        KMAX=args.kmax,
        seed=args.seed,
    )
    print(f"Collected {len(samples)} samples.")
    if not samples:
        raise RuntimeError("No samples were collected. Check data_root and parameters.")

    train_s, val_s = split_samples(samples, seed=args.seed)
    print(f"Train {len(train_s)} | Val {len(val_s)}")

    cnn = train_cnn_model(
        train_s, val_s, device=device, epochs=args.cnn_epochs, batch_size=args.cnn_batch, horizon=args.horizon
    )
    cnn_ADE, cnn_FDE = metrics.eval_cnn(cnn, val_s, device=device)
    cnn_NLL = metrics.nll_cnn_as_single_mode(cnn, val_s, device=device, max_cases=500, log_var_value=0.0)

    gnn = None
    gnn_ADE = gnn_FDE = float("nan")
    if not args.skip_gnn:
        gnn = train_gnn_model(
            train_s,
            val_s,
            device=device,
            epochs=args.gnn_epochs,
            batch_size=args.gnn_batch,
            hidden=args.hidden,
            horizon=args.horizon,
            seed=args.seed,
        )
        gnn_ADE, gnn_FDE = metrics.eval_gnn_single(gnn, val_s, device=device)

    gnn_multi = None
    gnn_multi_NLL = float("nan")
    if not args.skip_gnn_multi:
        gnn_multi = train_gnn_multi_model(
            train_s,
            val_s,
            device=device,
            epochs=args.gnn_multi_epochs,
            batch_size=args.gnn_multi_batch,
            hidden=args.hidden,
            horizon=args.horizon,
            seed=args.seed,
            k_modes=args.k_modes,
        )
        gnn_multi_NLL = metrics.nll_gnn_multi(gnn_multi, val_s, device=device, max_cases=500)

    print("\n==== Validation metrics ====")
    print(f"CNN (target-only): ADE={cnn_ADE:.3f} | FDE={cnn_FDE:.3f} | NLL={cnn_NLL:.3f}")
    if gnn is not None:
        print(f"GNN (single-mode): ADE={gnn_ADE:.3f} | FDE={gnn_FDE:.3f}")
    if gnn_multi is not None:
        print(f"GNN-Multi (K={config.K_MODES}): NLL={gnn_multi_NLL:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(cnn.state_dict(), os.path.join(args.save_dir, "cnn.pt"))
    if gnn is not None:
        torch.save(gnn.state_dict(), os.path.join(args.save_dir, "gnn.pt"))
    if gnn_multi is not None:
        torch.save(gnn_multi.state_dict(), os.path.join(args.save_dir, "gnn_multi.pt"))
    print(f"Saved checkpoints to {args.save_dir}")


if __name__ == "__main__":
    main()
