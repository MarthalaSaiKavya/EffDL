## Lyft motion forecasting – notebook split

This folder turns the `EffDL_FinalProject.ipynb` Colab notebook into importable Python modules and runnable scripts.

### Layout
- `config.py` – shared hyperparameters, device/seed helpers.
- `data_loading.py` – Zarr utilities and table loader.
- `sampling.py` – trajectory + graph sampling helpers.
- `datasets.py` – PyTorch datasets/loaders for the CNN baseline.
- `models.py` – CNN baseline, single-mode GNN, and multi-modal GNN.
- `metrics.py` – ADE/FDE/NLL metrics plus streaming evaluator.
- `plotting.py` – optional visualization utilities.
- `train.py` – end-to-end training for CNN, GNN, and GNN-Multi.
- `evaluate.py` – streaming evaluation across many scenes.
- `download_data.py` – Kaggle download helper.
- `requirements.txt` – Python dependencies.

### Setup
1) `cd EffDL_FinalProject_split`
2) (Optional) create a venv: `python3 -m venv .venv && source .venv/bin/activate`
3) Install deps: `pip install -r requirements.txt`
4) Configure Kaggle credentials (`~/.kaggle/kaggle.json` with `0600` perms).
5) Download the dataset (writes under `lyft_data/` by default):
   ```bash
   python3 download_data.py --data-root lyft_data
   ```
   If you already have the Zarr split, point `--data-root` (or `LYFT_DATA_ROOT`) to it.

### Train
Runs a compact version of the notebook pipeline (small defaults keep RAM reasonable; increase as needed).
```bash
python3 train.py \
  --data-root lyft_data \
  --max-samples 500 \
  --cnn-epochs 1 --gnn-epochs 1 --gnn-multi-epochs 1 \
  --history 10 --horizon 30 --kmax 24 \
  --cnn-batch 128 --gnn-batch 32 --gnn-multi-batch 16
```
Outputs checkpoints to `artifacts/` (`cnn.pt`, `gnn.pt`, `gnn_multi.pt`) plus console metrics (ADE/FDE/NLL).

Key flags:
- `--load-into-ram` / `--load-agents-into-ram` to force eager loading (default uses lazy Zarr to save memory).
- `--k-modes` to change the number of trajectories in the multi-modal head.
- `--max-samples`, `--frames-per-scene`, `--scenes` to control the sampled subset size.

### Streaming eval (100k-style sweep)
After training, stream metrics across many scenes without holding everything in RAM:
```bash
python3 evaluate.py \
  --data-root lyft_data \
  --max-samples 10000 \
  --samples-per-scene 20 \
  --scene-count 200 \
  --cnn-ckpt artifacts/cnn.pt \
  --gnn-ckpt artifacts/gnn.pt \
  --gnn-multi-ckpt artifacts/gnn_multi.pt
```

### Notes
- Defaults match the notebook safety settings (H=10, T=30, K=24 neighbors, K=3 modes). Adjust down if you hit OOM on CPU.
- Plotting helpers live in `plotting.py`; call them from a small script/notebook with a validation sample to reproduce the figures from the original notebook.
- If you change `--history`, `--horizon`, `--k-modes`, keep the same values for both training and evaluation.
