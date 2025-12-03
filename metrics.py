import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config
from sampling import collect_samples_from_single_scene


def ade_fde_batch(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float]:
    d = torch.linalg.norm(pred - gt, dim=-1)  # [B,T]
    ade = d.mean(dim=1).mean().item()
    fde = d[:, -1].mean().item()
    return ade, fde


def eval_cnn(cnn_model, samples: List[Dict], device: str = "cpu", max_cases: Optional[int] = None):
    cnn_model.eval()
    all_ade, all_fde, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, s in enumerate(samples):
            if max_cases is not None and i >= max_cases:
                break
            x = torch.tensor(s["x_hist"], dtype=torch.float32).t().unsqueeze(0).to(device)
            y = torch.tensor(s["x_fut"], dtype=torch.float32).unsqueeze(0).to(device)
            p = cnn_model(x)
            d = torch.linalg.norm(p - y, dim=-1)
            all_ade += d.mean().item()
            all_fde += d[0, -1].item()
            n += 1
    return all_ade / max(1, n), all_fde / max(1, n)


def eval_gnn_single(gnn_model, samples: List[Dict], device: str = "cpu", max_cases: Optional[int] = None):
    gnn_model.eval()
    all_ade, all_fde, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, s in enumerate(samples):
            if max_cases is not None and i >= max_cases:
                break
            X = torch.tensor(s["X"], dtype=torch.float32).to(device)
            EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
            EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
            y = torch.tensor(s["x_fut"], dtype=torch.float32).to(device)
            p = gnn_model(X, EI, EA, s["target_index"])
            d = torch.linalg.norm(p - y, dim=-1)
            all_ade += d.mean().item()
            all_fde += d[-1].item()
            n += 1
    return all_ade / max(1, n), all_fde / max(1, n)


def gaussian_log_prob(y, mu, log_var):
    """
    y:      [B, T, 2]
    mu:     [B, K, T, 2]
    log_var:[B, K]
    """
    B, T, _ = y.shape
    K = mu.shape[1]
    y_exp = y.unsqueeze(1).expand(-1, K, -1, -1)
    diff = y_exp - mu
    var = torch.exp(log_var).unsqueeze(-1).unsqueeze(-1)
    inv_var = 1.0 / var
    sq = (diff * diff * inv_var).sum(dim=(-1, -2))
    d = T * 2
    log_det = d * log_var
    log_norm = -0.5 * (sq + log_det + d * math.log(2 * math.pi))
    return log_norm


def mixture_nll(y, mu, log_var, logits):
    log_p_y_given_k = gaussian_log_prob(y, mu, log_var)
    log_pi = F.log_softmax(logits, dim=-1)
    log_mix = torch.logsumexp(log_pi + log_p_y_given_k, dim=-1)
    return -log_mix.mean()


def nll_cnn_as_single_mode(
    cnn_model, samples: List[Dict], device: str = "cpu", max_cases: Optional[int] = None, log_var_value: float = 0.0
):
    cnn_model.eval()
    losses: List[float] = []
    log_var = torch.tensor([[log_var_value]], device=device)
    logits = torch.zeros_like(log_var)
    with torch.no_grad():
        for i, s in enumerate(samples):
            if max_cases is not None and i >= max_cases:
                break
            x = torch.tensor(s["x_hist"], dtype=torch.float32).t().unsqueeze(0).to(device)
            y = torch.tensor(s["x_fut"], dtype=torch.float32).unsqueeze(0).to(device)
            p = cnn_model(x)
            mu = p.unsqueeze(1)
            loss = mixture_nll(y, mu, log_var, logits)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def nll_gnn_multi(gnn_model, samples: List[Dict], device: str = "cpu", max_cases: Optional[int] = None):
    gnn_model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for i, s in enumerate(samples):
            if max_cases is not None and i >= max_cases:
                break
            X = torch.tensor(s["X"], dtype=torch.float32).to(device)
            EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
            EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
            y = torch.tensor(s["x_fut"], dtype=torch.float32).unsqueeze(0).to(device)
            mu, log_var, logits = gnn_model(X, EI, EA, s["target_index"])
            losses.append(mixture_nll(y, mu, log_var, logits).item())
    return float(np.mean(losses)) if losses else float("nan")


def eval_gnn_single_safe(gnn_model, samples: List[Dict], device: str = "cpu", max_cases: Optional[int] = None):
    """Safe eval that skips degenerate graphs and NaNs."""
    gnn_model.eval()
    all_ade = all_fde = 0.0
    n_used = n_skipped = 0
    with torch.no_grad():
        for i, s in enumerate(samples):
            if max_cases is not None and i >= max_cases:
                break
            EI_np = s["edge_index"]
            if EI_np.shape[1] == 0:
                n_skipped += 1
                continue
            X = torch.tensor(s["X"], dtype=torch.float32).to(device)
            EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
            EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
            y = torch.tensor(s["x_fut"], dtype=torch.float32).to(device)
            p = gnn_model(X, EI, EA, s["target_index"])
            if torch.isnan(p).any():
                n_skipped += 1
                continue
            d = torch.linalg.norm(p - y, dim=-1)
            if torch.isnan(d).any() or torch.isinf(d).any():
                n_skipped += 1
                continue
            all_ade += d.mean().item()
            all_fde += d[-1].item()
            n_used += 1
    ade = all_ade / max(1, n_used)
    fde = all_fde / max(1, n_used)
    return ade, fde, n_used, n_skipped


def nll_gnn_multi_safe(gnn_model, samples: List[Dict], device: str = "cpu", max_cases: Optional[int] = None):
    losses: List[float] = []
    n_used = n_skipped = 0
    gnn_model.eval()
    with torch.no_grad():
        for i, s in enumerate(samples):
            if max_cases is not None and i >= max_cases:
                break
            if s["edge_index"].shape[1] == 0:
                n_skipped += 1
                continue
            X = torch.tensor(s["X"], dtype=torch.float32).to(device)
            EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
            EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
            y = torch.tensor(s["x_fut"], dtype=torch.float32).unsqueeze(0).to(device)
            mu, log_var, logits = gnn_model(X, EI, EA, s["target_index"])
            loss = mixture_nll(y, mu, log_var, logits)
            if torch.isnan(loss) or torch.isinf(loss):
                n_skipped += 1
                continue
            losses.append(loss.item())
            n_used += 1
    nll = float(np.mean(losses)) if losses else float("nan")
    return nll, n_used, n_skipped


def cnn_metrics_for_sample(cnn_model, s: Dict, device: str = "cpu"):
    x = torch.tensor(s["x_hist"], dtype=torch.float32).t().unsqueeze(0).to(device)
    y = torch.tensor(s["x_fut"], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        p = cnn_model(x)
    d = torch.linalg.norm(p - y, dim=-1)
    ade = d.mean().item()
    fde = d[0, -1].item()
    return ade, fde, p


def gnn_metrics_for_sample(gnn_model, s: Dict, device: str = "cpu"):
    EI_np = s["edge_index"]
    if EI_np.shape[1] == 0:
        return None
    X = torch.tensor(s["X"], dtype=torch.float32).to(device)
    EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
    EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
    y = torch.tensor(s["x_fut"], dtype=torch.float32).to(device)
    with torch.no_grad():
        p = gnn_model(X, EI, EA, s["target_index"])
    d = torch.linalg.norm(p - y, dim=-1)
    ade = d.mean().item()
    fde = d[-1].item()
    return ade, fde, p


def gnn_multi_nll_for_sample(gnn_multi_model, s: Dict, device: str = "cpu"):
    if s["edge_index"].shape[1] == 0:
        return None
    X = torch.tensor(s["X"], dtype=torch.float32).to(device)
    EI = torch.tensor(s["edge_index"], dtype=torch.long).to(device)
    EA = torch.tensor(s["edge_attr"], dtype=torch.float32).to(device)
    y = torch.tensor(s["x_fut"], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, log_var, logits = gnn_multi_model(X, EI, EA, s["target_index"])
        loss = mixture_nll(y, mu, log_var, logits)
    return loss.item()


def streaming_eval_over_scenes(
    tables,
    cnn_model,
    gnn_model,
    gnn_multi_model,
    max_global_samples: int = 100000,
    scenes_to_visit: Optional[Iterable[int]] = None,
    max_samples_per_scene: int = 30,
    device: str = "cpu",
):
    """
    Stream ADE/FDE/NLL over many scenes without storing everything in RAM.
    """
    if scenes_to_visit is None:
        scene_indices = np.arange(len(tables.scenes))
    else:
        scene_indices = np.array(list(scenes_to_visit), dtype=int)

    rng_local = np.random.default_rng(123)
    rng_local.shuffle(scene_indices)

    cnn_ade_sum = cnn_fde_sum = cnn_nll_sum = 0.0
    cnn_count = 0

    gnn_ade_sum = gnn_fde_sum = 0.0
    gnn_count = gnn_skipped = 0

    gnnm_nll_sum = 0.0
    gnnm_count = gnnm_skipped = 0

    total_used = 0

    for si in scene_indices:
        remaining = max_global_samples - total_used
        if remaining <= 0:
            break
        samples = collect_samples_from_single_scene(
            tables,
            scene_index=int(si),
            max_samples=min(max_samples_per_scene, remaining),
            H=config.SAFE_H,
            T=config.SAFE_T,
            KMAX=config.SAFE_KMAX,
        )
        for s in samples:
            if total_used >= max_global_samples:
                break

            ade_c, fde_c, _ = cnn_metrics_for_sample(cnn_model, s, device=device)
            nll_c = nll_cnn_as_single_mode(cnn_model, [s], device=device, log_var_value=0.0)
            cnn_ade_sum += ade_c
            cnn_fde_sum += fde_c
            cnn_nll_sum += nll_c
            cnn_count += 1

            gnn_res = gnn_metrics_for_sample(gnn_model, s, device=device)
            if gnn_res is None:
                gnn_skipped += 1
            else:
                ade_g, fde_g, _ = gnn_res
                gnn_ade_sum += ade_g
                gnn_fde_sum += fde_g
                gnn_count += 1

            gnnm_res = gnn_multi_nll_for_sample(gnn_multi_model, s, device=device)
            if gnnm_res is None:
                gnnm_skipped += 1
            else:
                gnnm_nll_sum += gnnm_res
                gnnm_count += 1

            total_used += 1

        if total_used >= max_global_samples:
            break

    return dict(
        total_used=total_used,
        cnn_ADE=cnn_ade_sum / max(1, cnn_count),
        cnn_FDE=cnn_fde_sum / max(1, cnn_count),
        cnn_NLL=cnn_nll_sum / max(1, cnn_count),
        gnn_ADE=gnn_ade_sum / max(1, gnn_count),
        gnn_FDE=gnn_fde_sum / max(1, gnn_count),
        gnn_skipped=gnn_skipped,
        gnn_multi_NLL=gnnm_nll_sum / max(1, gnnm_count),
        gnn_multi_skipped=gnnm_skipped,
    )
