import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_case(sample: Dict, pred, title: str):
    xh, xf = sample["x_hist"], sample["x_fut"]
    plt.figure(figsize=(5, 5))
    plt.plot(xh[:, 0], xh[:, 1], "-o", label="history", linewidth=1)
    plt.plot(xf[:, 0], xf[:, 1], "-o", label="future (GT)", linewidth=2)
    plt.plot(pred[:, 0], pred[:, 1], "-o", label="pred", linewidth=2)
    plt.gca().set_aspect("equal", "box")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def get_last_layer_attention(gnn, sample: Dict, device: str = "cpu"):
    """
    Returns attention from target node to all nodes (mean over heads).
    """
    gnn.eval()
    with torch.no_grad():
        X = torch.tensor(sample["X"], dtype=torch.float32).to(device)
        EI = torch.tensor(sample["edge_index"], dtype=torch.long).to(device)
        target_idx = sample["target_index"]

        Hn = gnn.xenc(X)
        for gat in gnn.layers[:-1]:
            Hn = torch.relu(Hn + gat(Hn, EI))

        gat = gnn.layers[-1]
        N = Hn.size(0)
        q = gat.q(Hn).view(N, gat.heads, -1)
        k = gat.k(Hn).view(N, gat.heads, -1)
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        mask[EI[0], EI[1]] = True
        scores = torch.einsum("ihd,jhd->hij", q, k) / np.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        attn_target = alpha[:, target_idx, :].mean(dim=0)
        return attn_target.cpu().numpy(), sample["X"].copy()


def plot_explainable_case(gnn, sample: Dict, device: str = "cpu", title_prefix: str = "GNN explainability"):
    attn, X_nodes = get_last_layer_attention(gnn, sample, device=device)
    node_xy = X_nodes[:, 0:2]
    xh = sample["x_hist"]
    xf = sample["x_fut"]

    plt.figure(figsize=(6, 6))
    plt.plot(xh[:, 0], xh[:, 1], "-o", label="history", linewidth=1)
    plt.plot(xf[:, 0], xf[:, 1], "-o", label="future", linewidth=2)

    attn_pos = np.maximum(attn, 0)
    attn_norm = attn_pos / attn_pos.max() if attn_pos.max() > 0 else np.zeros_like(attn_pos)
    sizes = 50 + 250 * attn_norm
    plt.scatter(node_xy[:, 0], node_xy[:, 1], s=sizes, c="tab:blue", alpha=0.6, label="neighbors")
    plt.scatter([0], [0], s=120, marker="*", c="gold", edgecolors="k", label="target")

    plt.gca().set_aspect("equal", "box")
    plt.grid(True)
    plt.xlabel("x_rel (m)")
    plt.ylabel("y_rel (m)")
    plt.title(f"{title_prefix} (node size ~ attention)")
    plt.legend()
    plt.show()


def plot_cnn_vs_gnn_case(cnn, gnn, sample: Dict, device: str = "cpu", title_prefix: str = "CNN vs GNN"):
    x_hist = sample["x_hist"]
    x_fut = sample["x_fut"]

    x_in = torch.tensor(x_hist, dtype=torch.float32).t().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_cnn = cnn(x_in).cpu().numpy()[0]

    X = torch.tensor(sample["X"], dtype=torch.float32).to(device)
    EI = torch.tensor(sample["edge_index"], dtype=torch.long).to(device)
    EA = torch.tensor(sample["edge_attr"], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_gnn = gnn(X, EI, EA, sample["target_index"]).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(x_hist[:, 0], x_hist[:, 1], "-o", label="history", linewidth=1)
    plt.plot(x_fut[:, 0], x_fut[:, 1], "-o", label="future (GT)", linewidth=2)
    plt.plot(pred_cnn[:, 0], pred_cnn[:, 1], "-o", label="CNN", linewidth=2)
    plt.plot(pred_gnn[:, 0], pred_gnn[:, 1], "-o", label="GNN", linewidth=2)
    plt.gca().set_aspect("equal", "box")
    plt.grid(True)
    plt.xlabel("x_rel (m)")
    plt.ylabel("y_rel (m)")
    plt.title(title_prefix)
    plt.legend()
    plt.show()


def plot_multimodal_case(gnn_multi, sample: Dict, device: str = "cpu", title_prefix: str = "GNN multi-modal"):
    xh = sample["x_hist"]
    xf = sample["x_fut"]
    X = torch.tensor(sample["X"], dtype=torch.float32).to(device)
    EI = torch.tensor(sample["edge_index"], dtype=torch.long).to(device)
    EA = torch.tensor(sample["edge_attr"], dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, log_var, logits = gnn_multi(X, EI, EA, sample["target_index"])
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    mu_np = mu[0].cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(xh[:, 0], xh[:, 1], "-o", label="history", linewidth=1)
    plt.plot(xf[:, 0], xf[:, 1], "-o", label="future (GT)", linewidth=2)
    for k in range(mu_np.shape[0]):
        plt.plot(mu_np[k, :, 0], mu_np[k, :, 1], "-o", label=f"mode {k} (p={probs[k]:.2f})", linewidth=2)
    plt.gca().set_aspect("equal", "box")
    plt.grid(True)
    plt.xlabel("x_rel (m)")
    plt.ylabel("y_rel (m)")
    plt.title(f"{title_prefix} – K={mu_np.shape[0]}")
    plt.legend()
    plt.show()


def save_cnn_vs_gnn_plot(cnn, gnn, sample: Dict, fname: str, device: str = "cpu"):
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    x_hist = sample["x_hist"]
    x_fut = sample["x_fut"]
    x_in = torch.tensor(x_hist, dtype=torch.float32).t().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_cnn = cnn(x_in).cpu().numpy()[0]
    X = torch.tensor(sample["X"], dtype=torch.float32).to(device)
    EI = torch.tensor(sample["edge_index"], dtype=torch.long).to(device)
    EA = torch.tensor(sample["edge_attr"], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_gnn = gnn(X, EI, EA, sample["target_index"]).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(x_hist[:, 0], x_hist[:, 1], "-o", label="history", linewidth=1)
    plt.plot(x_fut[:, 0], x_fut[:, 1], "-o", label="future (GT)", linewidth=2)
    plt.plot(pred_cnn[:, 0], pred_cnn[:, 1], "-o", label="CNN", linewidth=2)
    plt.plot(pred_gnn[:, 0], pred_gnn[:, 1], "-o", label="GNN", linewidth=2)
    plt.gca().set_aspect("equal", "box")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x_rel (m)")
    plt.ylabel("y_rel (m)")
    plt.title("CNN vs GNN")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()


def save_multimodal_plot(gnn_multi, sample: Dict, fname: str, device: str = "cpu"):
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    x_hist = sample["x_hist"]
    x_fut = sample["x_fut"]
    X = torch.tensor(sample["X"], dtype=torch.float32).to(device)
    EI = torch.tensor(sample["edge_index"], dtype=torch.long).to(device)
    EA = torch.tensor(sample["edge_attr"], dtype=torch.float32).to(device)
    with torch.no_grad():
        mu, log_var, logits = gnn_multi(X, EI, EA, sample["target_index"])
    K = mu.shape[1]
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    mu_np = mu[0].cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(x_hist[:, 0], x_hist[:, 1], "-o", label="history", linewidth=1)
    plt.plot(x_fut[:, 0], x_fut[:, 1], "-o", label="future (GT)", linewidth=2)
    for k in range(K):
        plt.plot(mu_np[k, :, 0], mu_np[k, :, 1], "-o", label=f"mode {k} (p={probs[k]:.2f})", linewidth=2)
    plt.gca().set_aspect("equal", "box")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x_rel (m)")
    plt.ylabel("y_rel (m)")
    plt.title("GNN multi-modal")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
