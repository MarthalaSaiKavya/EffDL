import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numcodecs import get_codec

import config


@dataclass
class LyftTables:
    """Container for the Lyft Zarr tables."""

    zarr_root: Path
    scenes: Any
    frames: Any
    agents: Any
    traffic_light_faces: Any = None


def _load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def _np_dtype_from_v2(dtype_field):
    """Lyft stores structured dtypes in Zarr v2 metadata."""
    if isinstance(dtype_field, str):
        return np.dtype(dtype_field)
    fields = []
    for f in dtype_field:
        name = f[0]
        base = f[1] if len(f) > 1 else "?"
        shape = tuple(f[2]) if len(f) > 2 else ()
        fields.append((name, np.dtype(base), shape))
    return np.dtype(fields)


def read_first_chunk(arr_dir: Path):
    """Decode the first chunk of a Zarr v2 array into a numpy array."""
    meta = _load_json(arr_dir / ".zarray")
    shape = tuple(meta["shape"])
    chunks = tuple(meta["chunks"])
    dtype = _np_dtype_from_v2(meta["dtype"])
    compressor = meta.get("compressor")
    filters = meta.get("filters") or []

    key = ".".join(["0"] * len(chunks)) if len(chunks) > 1 else "0"
    with open(arr_dir / key, "rb") as f:
        buf = f.read()
    if compressor:
        buf = get_codec(compressor).decode(buf)
    for flt in filters:
        buf = get_codec(flt).decode(buf)

    chunk_shape = tuple(min(c, s) for c, s in zip(chunks, shape))
    arr = np.frombuffer(buf, dtype=dtype, count=int(np.prod(chunk_shape)))
    return arr.reshape(-1)


def _read_all(arr_dir: Path):
    """Load a full Zarr v2 array into memory (use only for smaller tables)."""
    meta = _load_json(arr_dir / ".zarray")
    shape = tuple(meta["shape"])
    chunks = tuple(meta["chunks"])
    dtype = _np_dtype_from_v2(meta["dtype"])
    comp = meta.get("compressor")
    filts = meta.get("filters") or []

    arr = np.empty(shape, dtype=dtype)

    from itertools import product

    grid = [range((s + c - 1) // c) for s, c in zip(shape, chunks)]
    for idx in product(*grid):
        key = ".".join(str(i) for i in idx)
        with open(arr_dir / key, "rb") as f:
            buf = f.read()
        if comp:
            buf = get_codec(comp).decode(buf)
        for flt in filts:
            buf = get_codec(flt).decode(buf)

        chunk_shape = tuple(min(c, s - i * c) for i, c, s in zip(idx, chunks, shape))
        block = np.frombuffer(buf, dtype=dtype, count=int(np.prod(chunk_shape))).reshape(chunk_shape)
        slices = tuple(slice(i * c, i * c + cs) for i, c, cs in zip(idx, chunks, chunk_shape))
        arr[slices] = block
    return arr


def _nice_subnames(field, shape):
    if shape == (2,) and field == "centroid":
        return ["x", "y"]
    if shape == (2,) and field == "velocity":
        return ["vx", "vy"]
    if shape == (3,) and field == "ego_translation":
        return ["x", "y", "z"]
    if shape == (3, 3) and field == "ego_rotation":
        return [f"r{i}{j}" for i in range(3) for j in range(3)]
    if field == "label_probabilities":
        return [f"lp_{i}" for i in range(int(np.prod(shape)))]
    return [str(i) for i in range(int(np.prod(shape)))]


def struct_rows_to_df(arr, n=5):
    """Flatten the first n rows of a structured array into a dataframe."""
    import pandas as pd

    n = min(n, arr.shape[0])
    names = arr.dtype.names or []
    cols = {}
    for name in names:
        a = np.asarray(arr[name][:n])
        if a.ndim == 1:
            cols[name] = a
        else:
            trailing = a.shape[1:]
            flat = a.reshape(n, int(np.prod(trailing)))
            suf = _nice_subnames(name, trailing)
            for j in range(flat.shape[1]):
                cols[f"{name}_{suf[j] if j < len(suf) else j}"] = flat[:, j]
    return pd.DataFrame(cols)


def choose_zarr_root(data_root: Path = config.DATA_ROOT) -> Optional[Path]:
    """Pick the first available split under the given data root."""
    for cand in config.ZARR_CANDIDATES:
        z = data_root / cand
        if z.exists():
            return z
    return None


def open_tables(
    data_root: Path = config.DATA_ROOT,
    prefer_lazy: bool = True,
    load_agents_into_ram: bool = False,
) -> LyftTables:
    """
    Open the Lyft Level 5 Zarr tables.

    prefer_lazy: use zarr arrays (keeps RAM low).
    load_agents_into_ram: if False, agents is left as a lazy zarr array even when prefer_lazy=False.
    """
    zarr_root = choose_zarr_root(data_root)
    if zarr_root is None:
        raise FileNotFoundError(
            f"No Zarr split found under {data_root}. Expected one of: {', '.join(config.ZARR_CANDIDATES)}"
        )

    if prefer_lazy:
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("Install zarr/numcodecs (see requirements.txt) or set prefer_lazy=False.") from exc
        root = zarr.open(str(zarr_root), mode="r")
        scenes = root["scenes"]
        frames = root["frames"]
        agents = root["agents"]
        tl_faces = root.get("traffic_light_faces")
        return LyftTables(zarr_root, scenes, frames, agents, tl_faces)

    scenes = _read_all(zarr_root / "scenes")
    frames = _read_all(zarr_root / "frames")

    if load_agents_into_ram:
        agents = _read_all(zarr_root / "agents")
    else:
        import zarr

        agents = zarr.open(str(zarr_root / "agents"), mode="r")

    tl_dir = zarr_root / "traffic_light_faces"
    tl_faces = _read_all(tl_dir) if tl_dir.exists() else None

    return LyftTables(zarr_root, scenes, frames, agents, tl_faces)
