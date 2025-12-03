import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class CNN1D(nn.Module):
    """Target-only 1D CNN baseline."""

    def __init__(self, in_ch: int = 2, T_out: int = config.SAFE_T, hidden: int = 96):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(in_ch, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, hidden, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, T_out * 2),
        )
        self.T_out = T_out

    def forward(self, x):
        h = self.fe(x)
        y = self.head(h).view(-1, self.T_out, 2)
        return y


class NodeEnc(nn.Module):
    def __init__(self, d_in: int = 8, h: int = config.SAFE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class EdgeEnc(nn.Module):
    def __init__(self, d_in: int = 4, h: int = config.SAFE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, e):
        return self.net(e)


class MaskedGAT(nn.Module):
    """Lightweight multi-head attention with an explicit edge mask."""

    def __init__(self, h: int = config.SAFE_HIDDEN, heads: int = 4):
        super().__init__()
        self.h = h
        self.heads = heads
        self.q = nn.Linear(h, h * heads, bias=False)
        self.k = nn.Linear(h, h * heads, bias=False)
        self.v = nn.Linear(h, h * heads, bias=False)
        self.proj = nn.Linear(h * heads, h)

    def forward(self, H, edge_index):
        N = H.size(0)
        q = self.q(H).view(N, self.heads, -1)
        k = self.k(H).view(N, self.heads, -1)
        v = self.v(H).view(N, self.heads, -1)

        mask = torch.zeros(N, N, dtype=torch.bool, device=H.device)
        mask[edge_index[0], edge_index[1]] = True

        scores = torch.einsum("nhd,mhd->hnm", q, k) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        out = torch.einsum("hnm,mhd->nhd", alpha, v).reshape(N, -1)
        return self.proj(out)


class TrajHead(nn.Module):
    """Single-mode trajectory decoder."""

    def __init__(self, h: int = config.SAFE_HIDDEN, T: int = config.SAFE_T):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, T * 2))
        self.T = T

    def forward(self, h_tgt):
        return self.fc(h_tgt).view(-1, 2)


class GNNMotion(nn.Module):
    """Interaction-aware GNN (single-mode output)."""

    def __init__(
        self,
        x_dim: int = 8,
        e_dim: int = 4,
        h: int = config.SAFE_HIDDEN,
        heads: int = 4,
        layers: int = 2,
        T: int = config.SAFE_T,
    ):
        super().__init__()
        self.xenc = NodeEnc(x_dim, h)
        self.eenc = EdgeEnc(e_dim, h)
        self.layers = nn.ModuleList([MaskedGAT(h, heads) for _ in range(layers)])
        self.dec = TrajHead(h, T)

    def forward(self, X, EI, EA, target_index):
        Hn = self.xenc(X)
        _ = self.eenc(EA)  # placeholder for future use
        for gat in self.layers:
            Hn = F.relu(Hn + gat(Hn, EI))
        return self.dec(Hn[target_index])


class MultiModalHead(nn.Module):
    """Predict K candidate futures + logits + log variances."""

    def __init__(self, h: int = config.SAFE_HIDDEN, T: int = config.SAFE_T, K: int = config.K_MODES):
        super().__init__()
        self.T = T
        self.K = K
        self.mlp = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, K * T * 2 + K * 2))

    def forward(self, h_tgt) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          mu:      [B, K, T, 2]
          log_var: [B, K]
          logits:  [B, K]
        """
        out = self.mlp(h_tgt)
        B = out.size(0)
        num_mu = self.K * self.T * 2
        num_logits = self.K

        mu_flat = out[:, :num_mu]
        logits = out[:, num_mu : num_mu + num_logits]
        log_var = out[:, num_mu + num_logits :]

        mu = mu_flat.view(B, self.K, self.T, 2)
        return mu, log_var, logits


class GNNMultiMotion(nn.Module):
    """Interaction-aware GNN with mixture likelihood (K modes)."""

    def __init__(
        self,
        x_dim: int = 8,
        e_dim: int = 4,
        h: int = config.SAFE_HIDDEN,
        heads: int = 4,
        layers: int = 2,
        T: int = config.SAFE_T,
        K: int = config.K_MODES,
    ):
        super().__init__()
        self.xenc = NodeEnc(x_dim, h)
        self.eenc = EdgeEnc(e_dim, h)
        self.layers = nn.ModuleList([MaskedGAT(h, heads) for _ in range(layers)])
        self.dec = MultiModalHead(h=h, T=T, K=K)

    def forward(self, X, EI, EA, target_index):
        Hn = self.xenc(X)
        _ = self.eenc(EA)
        for gat in self.layers:
            Hn = F.relu(Hn + gat(Hn, EI))
        h_tgt = Hn[target_index].unsqueeze(0)
        mu, log_var, logits = self.dec(h_tgt)
        return mu, log_var, logits
