import numpy as np
import torch


def normalize_time(times, t_min=None, t_max=None):
    if t_min is None:
        t_min = float(np.min(times))
    if t_max is None:
        t_max = float(np.max(times))
    if t_max == t_min:
        return np.zeros_like(times, dtype=float), t_min, t_max
    return (times - t_min) / (t_max - t_min), t_min, t_max


def build_dataset(df, qubit, pauli, normalize_time_flag=True, t_min=None, t_max=None):
    data = df[(df["qubit"] == qubit) & (df["pauli"] == pauli)].copy()
    times = data["time"].to_numpy(dtype=float)
    if normalize_time_flag:
        times, t_min, t_max = normalize_time(times, t_min=t_min, t_max=t_max)
    else:
        t_min = float(np.min(times)) if len(times) else 0.0
        t_max = float(np.max(times)) if len(times) else 0.0
    outcomes = data["outcome"].to_numpy(dtype=float)
    y01 = ((outcomes + 1.0) / 2.0).astype(np.float32)
    train_x = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
    train_y = torch.tensor(y01, dtype=torch.float32)
    return train_x, train_y, data, t_min, t_max


def print_dataset_diagnostics(data, label="dataset"):
    if data.empty:
        print(f"[{label}] empty dataset")
        return
    n = len(data)
    unique_times = data["time"].nunique()
    frac_pos = float(((data["outcome"] + 1.0) / 2.0).mean())
    t_min = float(data["time"].min())
    t_max = float(data["time"].max())
    print(f"[{label}] rows={n}, unique_times={unique_times}, frac_pos={frac_pos:.3f}, t_min={t_min:.4f}, t_max={t_max:.4f}")


def compute_empirical_p(data):
    grouped = data.groupby("time")["outcome"]
    n = grouped.size()
    k = grouped.apply(lambda s: (s == 1).sum())
    p_hat = (k / n).to_numpy()
    times = n.index.to_numpy(dtype=float)
    return times, p_hat


def aggregate_by_time(data):
    grouped = data.groupby("time")["outcome"]
    n = grouped.size().to_numpy(dtype=float)
    k = grouped.apply(lambda s: (s == 1).sum()).to_numpy(dtype=float)
    p_hat = k / n
    var = p_hat * (1.0 - p_hat) / np.maximum(n, 1.0)
    times = grouped.size().index.to_numpy(dtype=float)
    return times, p_hat.astype(float), var.astype(float)


def make_inducing_points(train_x, num_inducing):
    if train_x.numel() == 0:
        return train_x
    xs = torch.sort(train_x.squeeze(-1))[0]
    xs_unique = torch.unique(xs)
    if len(xs_unique) <= num_inducing:
        return xs_unique.unsqueeze(-1).clone()
    idx = torch.linspace(0, len(xs_unique) - 1, steps=num_inducing).long()
    return xs_unique[idx].unsqueeze(-1).clone()


def mae_against_true(t_test, pred_prob, t_true, true_prob):
    if len(t_test) == 0:
        return float("nan")
    pred = np.interp(t_true, t_test, pred_prob)
    return float(np.mean(np.abs(pred - true_prob)))
