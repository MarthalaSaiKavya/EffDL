import torch
from torch.utils.data import DataLoader, Dataset


class TargetOnlyDataset(Dataset):
    """Torch dataset that serves (history, future) pairs for the CNN baseline."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s = self.items[idx]
        x = torch.tensor(s["x_hist"], dtype=torch.float32).t()  # [2, H]
        y = torch.tensor(s["x_fut"], dtype=torch.float32)  # [T, 2]
        return x, y


def make_target_only_loaders(train_items, val_items, batch_size: int, drop_last_train: bool = True):
    dl_tr = DataLoader(
        TargetOnlyDataset(train_items),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
    )
    dl_va = DataLoader(
        TargetOnlyDataset(val_items),
        batch_size=batch_size,
        shuffle=False,
    )
    return dl_tr, dl_va
