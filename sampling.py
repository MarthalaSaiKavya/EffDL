from typing import Dict, Iterable, List, Optional

import numpy as np

import config
from config import set_seeds
from data_loading import LyftTables


def get_frame_agent_interval(tables: LyftTables, fi: int):
    row = tables.frames[fi]
    a0 = int(row["agent_index_interval"][0])
    a1 = int(row["agent_index_interval"][1])
    return a0, a1


def get_scene_frame_range(tables: LyftTables, si: int):
    row = tables.scenes[si]
    f0 = int(row["frame_index_interval"][0])
    f1 = int(row["frame_index_interval"][1])
    return range(f0, f1)


def get_agent_fields(tables: LyftTables, row_idx: int):
    r = tables.agents[row_idx]
    c = np.array(r["centroid"], dtype=np.float32)
    v = np.array(r["velocity"], dtype=np.float32)
    yaw = float(r["yaw"])
    ext = np.array(r["extent"], dtype=np.float32)
    return c, v, yaw, ext


def find_row_for_track_in_frame(tables: LyftTables, track_id_int: int, fi: int):
    a0, a1 = get_frame_agent_interval(tables, fi)
    if a1 <= a0:
        return None
    tids = np.array(tables.agents["track_id"][a0:a1], dtype=np.int64)
    m = np.nonzero(tids == track_id_int)[0]
    if m.size == 0:
        return None
    return a0 + int(m[0])


def neighbor_rows(tables: LyftTables, fi: int, anchor_xy, radius: float, kmax: int):
    a0, a1 = get_frame_agent_interval(tables, fi)
    if a1 <= a0:
        return []
    XY = np.array(tables.agents["centroid"][a0:a1], dtype=np.float32)
    d = np.linalg.norm(XY - anchor_xy[None, :], axis=1)
    idx = np.nonzero(d < radius)[0]
    if idx.size == 0:
        return []
    if idx.size > kmax:
        top = np.argpartition(d[idx], kmax - 1)[:kmax]
        idx = idx[top]
    return [a0 + int(k) for k in idx]


def build_sample(
    tables: LyftTables,
    target_row_t0: int,
    t0: int,
    H: int = config.SAFE_H,
    T: int = config.SAFE_T,
    R: float = 30.0,
    KMAX: int = config.SAFE_KMAX,
) -> Optional[Dict]:
    """Build a single trajectory + graph sample centered at target_row_t0 in frame t0."""
    tid = int(tables.agents["track_id"][target_row_t0])
    c0, v0, yaw0, ext0 = get_agent_fields(tables, target_row_t0)

    t_hist = list(range(t0 - H, t0))
    t_fut = list(range(t0, t0 + T))
    if t_hist[0] < 0 or t_fut[-1] >= len(tables.frames):
        return None

    tr = {}
    for fi in t_hist + t_fut:
        r = find_row_for_track_in_frame(tables, tid, fi)
        if r is None:
            return None
        tr[fi] = r

    xh, xf = [], []
    for fi in t_hist:
        c, *_ = get_agent_fields(tables, tr[fi])
        xh.append(c - c0)
    for fi in t_fut:
        c, *_ = get_agent_fields(tables, tr[fi])
        xf.append(c - c0)
    x_hist = np.stack(xh, 0).astype(np.float32)
    x_fut = np.stack(xf, 0).astype(np.float32)

    rows = neighbor_rows(tables, t0, c0, R, KMAX)
    if target_row_t0 in rows:
        rows.remove(target_row_t0)
    rows = [target_row_t0] + rows

    feats, cvs = [], []
    for r in rows:
        c, v, yaw, ext = get_agent_fields(tables, r)
        cvs.append((c, v))
        feats.append(np.concatenate([(c - c0), v, [yaw - yaw0], ext], 0))
    X = np.stack(feats, 0).astype(np.float32)

    N = X.shape[0]
    send, recv, eattr = [], [], []
    for i in range(N):
        ci, vi = cvs[i]
        for j in range(N):
            if i == j:
                continue
            cj, vj = cvs[j]
            rel = np.concatenate([(cj - ci), (vj - vi)], 0).astype(np.float32)
            send.append(i)
            recv.append(j)
            eattr.append(rel)

    edge_index = np.stack([np.array(send), np.array(recv)], 0).astype(np.int64)
    edge_attr = np.stack(eattr, 0) if eattr else np.zeros((0, 4), np.float32)

    return dict(
        x_hist=x_hist,
        x_fut=x_fut,
        X=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_index=0,
        c0=c0,
        t0=t0,
        target_tid=tid,
    )


def collect_samples(
    tables: LyftTables,
    max_samples: int = config.SAFE_MAX_SAMPLES,
    scenes_to_use: Optional[int] = None,
    frames_per_scene: int = config.SAFE_FRAMES_PER_SC,
    H: int = config.SAFE_H,
    T: int = config.SAFE_T,
    R: float = 30.0,
    KMAX: int = config.SAFE_KMAX,
    seed: int = 123,
) -> List[Dict]:
    """Fast sampler across a handful of scenes."""
    items: List[Dict] = []
    rng = np.random.default_rng(seed)

    if scenes_to_use is None:
        scenes_to_use = min(4, len(tables.scenes))

    for s in range(scenes_to_use):
        fr_rng = list(get_scene_frame_range(tables, s))
        if not fr_rng:
            continue
        stride = max(1, len(fr_rng) // frames_per_scene) if frames_per_scene else 1
        for t0 in fr_rng[::stride]:
            a0, a1 = get_frame_agent_interval(tables, t0)
            if a1 <= a0:
                continue
            rows = np.arange(a0, a1)
            if len(rows) > 10:
                rows = rng.choice(rows, size=10, replace=False)
            for r0 in rows:
                sd = build_sample(tables, int(r0), t0, H=H, T=T, R=R, KMAX=KMAX)
                if sd is not None:
                    items.append(sd)
            if len(items) >= max_samples:
                break
        if len(items) >= max_samples:
            break
    return items


def build_ram_safe_sample(
    tables: LyftTables,
    max_samples: int,
    n_scenes: int,
    frames_per_scene_max: int,
    agents_per_frame_max: int,
    H: int = config.SAFE_H,
    T: int = config.SAFE_T,
    R: float = 30.0,
    KMAX: int = config.SAFE_KMAX,
    seed: int = 42,
) -> List[Dict]:
    """
    Randomly subsample scenes/frames/agents across the full dataset without loading everything.
    """
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    total_scenes = len(tables.scenes)
    n_scenes = min(n_scenes, total_scenes)
    scene_indices = rng.choice(total_scenes, size=n_scenes, replace=False)

    items: List[Dict] = []
    for si in scene_indices:
        fr_rng = list(get_scene_frame_range(tables, int(si)))
        if not fr_rng:
            continue

        frame_indices = (
            rng.choice(fr_rng, size=frames_per_scene_max, replace=False)
            if len(fr_rng) > frames_per_scene_max
            else fr_rng
        )

        for t0 in frame_indices:
            a0, a1 = get_frame_agent_interval(tables, int(t0))
            if a1 <= a0:
                continue

            agent_rows = np.arange(a0, a1)
            if len(agent_rows) > agents_per_frame_max:
                agent_rows = rng.choice(agent_rows, size=agents_per_frame_max, replace=False)

            for r0 in agent_rows:
                sd = build_sample(
                    tables, int(r0), int(t0), H=H, T=T, R=R, KMAX=KMAX
                )
                if sd is not None:
                    items.append(sd)
                if len(items) >= max_samples:
                    return items
    return items


def collect_samples_from_single_scene(
    tables: LyftTables,
    scene_index: int,
    max_samples: int = 50,
    H: int = config.SAFE_H,
    T: int = config.SAFE_T,
    KMAX: int = config.SAFE_KMAX,
    radius: float = 30.0,
) -> List[Dict]:
    """Gather up to max_samples from one scene (used for streaming eval)."""
    out: List[Dict] = []
    fr_rng = list(get_scene_frame_range(tables, scene_index))
    if len(fr_rng) == 0:
        return out

    local_rng = np.random.default_rng(scene_index)
    local_rng.shuffle(fr_rng)

    for t0 in fr_rng:
        if len(out) >= max_samples:
            break
        a0, a1 = get_frame_agent_interval(tables, t0)
        if a1 <= a0:
            continue
        agent_rows = np.arange(a0, a1)
        if len(agent_rows) > 8:
            agent_rows = local_rng.choice(agent_rows, size=8, replace=False)
        for r0 in agent_rows:
            if len(out) >= max_samples:
                break
            s = build_sample(tables, int(r0), t0, H=H, T=T, R=radius, KMAX=KMAX)
            if s is not None:
                out.append(s)
    return out


def pick_non_degenerate_sample(samples: Iterable[Dict]) -> Optional[Dict]:
    """Return the first sample that has at least one edge."""
    for s in samples:
        if s["edge_index"].shape[1] > 0:
            return s
    return None
