"""
GLOBEM data preparation for autoresearch experiments.
Loads mobile sensing data, constructs 28-day windows for depression prediction.

Expected data layout in DATA_DIR:
    INS-W_1/  (or any subdirectory per dataset)
        FeatureData/
            rapids.csv          # columns: pid, date, feature1, feature2, ...
        SurveyData/
            dep_weekly.csv      # columns: pid, date, dep_endtotal

Usage:
    python prepare.py                    # prepare with default DATA_DIR
    python prepare.py --data-dir PATH    # custom data directory

Cached tensors are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import argparse
import glob

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify — imported by train.py)
# ---------------------------------------------------------------------------

WINDOW_DAYS = 28          # days in each input window
NUM_CLASSES = 2           # binary: depressed / not depressed
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
DEP_THRESHOLD = 2         # PHQ-4 score > threshold = positive (depressed)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.expanduser("~/data/globem")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TEST_RATIO = 0.2          # per-user train/test split ratio

# Will be set after data loading
N_FEATURES = None         # number of features per day (set by load_and_cache)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_datasets(data_dir):
    """Find dataset subdirectories that contain FeatureData/rapids.csv."""
    datasets = []
    for entry in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, entry)
        if not os.path.isdir(path):
            continue
        rapids = os.path.join(path, "FeatureData", "rapids.csv")
        if os.path.exists(rapids):
            datasets.append(entry)
    if not datasets:
        # Try looking for rapids.csv directly in data_dir
        direct = os.path.join(data_dir, "rapids.csv")
        if os.path.exists(direct):
            datasets.append(".")
    return datasets


def load_features(data_dir, dataset_name):
    """Load feature CSV for a dataset. Returns DataFrame with pid, date, features."""
    if dataset_name == ".":
        path = os.path.join(data_dir, "rapids.csv")
    else:
        path = os.path.join(data_dir, dataset_name, "FeatureData", "rapids.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["pid"] = df["pid"].astype(str)
    df["dataset"] = dataset_name
    return df


def load_labels(data_dir, dataset_name):
    """Load depression labels for a dataset. Returns DataFrame with pid, date, label."""
    if dataset_name == ".":
        path = os.path.join(data_dir, "dep_weekly.csv")
    else:
        path = os.path.join(data_dir, dataset_name, "SurveyData", "dep_weekly.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["pid"] = df["pid"].astype(str)
    # Binarize: score > threshold = depressed (1), else not (0)
    df["label"] = (df["dep_endtotal"] > DEP_THRESHOLD).astype(int)
    df["dataset"] = dataset_name
    return df[["pid", "date", "label", "dataset"]]


def construct_windows(features_df, labels_df, window_days=WINDOW_DAYS):
    """
    Construct (window_days, N_features) windows aligned to label dates.

    For each label row (pid, date), we take the window_days days ending on that date
    from the feature data. If fewer than window_days // 2 days are available, skip.

    Returns:
        X: np.ndarray of shape (N_samples, window_days, N_features)
        y: np.ndarray of shape (N_samples,) int
        pids: list of str (participant IDs, for splitting)
    """
    # Identify feature columns (everything except pid, date, dataset)
    meta_cols = {"pid", "date", "dataset"}
    feature_cols = sorted([c for c in features_df.columns if c not in meta_cols])
    n_features = len(feature_cols)

    # Index features by (pid, date)
    features_df = features_df.sort_values(["pid", "date"])

    X_list = []
    y_list = []
    pid_list = []

    for _, label_row in labels_df.iterrows():
        pid = label_row["pid"]
        end_date = label_row["date"]
        label = label_row["label"]
        dataset = label_row["dataset"]

        # Get this participant's features from the same dataset
        mask = (features_df["pid"] == pid) & (features_df["dataset"] == dataset)
        pid_features = features_df.loc[mask]
        if pid_features.empty:
            continue

        # Window: (end_date - window_days + 1) to end_date inclusive
        start_date = end_date - pd.Timedelta(days=window_days - 1)
        window_mask = (pid_features["date"] >= start_date) & (pid_features["date"] <= end_date)
        window = pid_features.loc[window_mask, feature_cols]

        if len(window) < window_days // 2:
            continue  # too much missing data

        # Create full window with NaN for missing days
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        pid_indexed = pid_features.set_index("date")
        full_window = pid_indexed.reindex(date_range)[feature_cols]

        # Forward fill then backward fill missing days, then fill remaining with 0
        full_window = full_window.ffill().bfill().fillna(0)

        X_list.append(full_window.values.astype(np.float32))
        y_list.append(label)
        pid_list.append(f"{dataset}_{pid}")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y, pid_list, feature_cols


def split_by_user(X, y, pids, test_ratio=TEST_RATIO, seed=42):
    """80/20 per-user split. All windows from a user go to either train or test."""
    rng = np.random.RandomState(seed)
    unique_pids = sorted(set(pids))
    rng.shuffle(unique_pids)

    n_test = max(1, int(len(unique_pids) * test_ratio))
    test_pids = set(unique_pids[:n_test])

    pid_arr = np.array(pids)
    test_mask = np.array([p in test_pids for p in pid_arr])
    train_mask = ~test_mask

    return (X[train_mask], y[train_mask]), (X[test_mask], y[test_mask])


def compute_norm_stats(X_train):
    """Compute per-feature mean and std from training data."""
    # X_train: (N, 28, F)
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X, mean, std):
    """Z-score normalize."""
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def get_cache_paths():
    return {
        "train_X": os.path.join(CACHE_DIR, "train_X.pt"),
        "train_y": os.path.join(CACHE_DIR, "train_y.pt"),
        "test_X": os.path.join(CACHE_DIR, "test_X.pt"),
        "test_y": os.path.join(CACHE_DIR, "test_y.pt"),
        "norm_mean": os.path.join(CACHE_DIR, "norm_mean.pt"),
        "norm_std": os.path.join(CACHE_DIR, "norm_std.pt"),
        "meta": os.path.join(CACHE_DIR, "meta.pt"),
    }


def cache_exists():
    return all(os.path.exists(p) for p in get_cache_paths().values())


def load_and_cache(data_dir=DATA_DIR):
    """Load GLOBEM data, process, and cache as .pt files."""
    global N_FEATURES
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = get_cache_paths()

    if cache_exists():
        meta = torch.load(paths["meta"], weights_only=True)
        N_FEATURES = int(meta["n_features"])
        print(f"Data: cached tensors found at {CACHE_DIR} (n_features={N_FEATURES})")
        return

    print(f"Data: loading GLOBEM from {data_dir}...")
    datasets = find_datasets(data_dir)
    if not datasets:
        print(f"Error: no datasets found in {data_dir}")
        print("Expected: subdirectories with FeatureData/rapids.csv and SurveyData/dep_weekly.csv")
        sys.exit(1)
    print(f"  Found datasets: {datasets}")

    # Load all datasets
    all_features = []
    all_labels = []
    for ds in datasets:
        print(f"  Loading {ds}...")
        feat = load_features(data_dir, ds)
        lab = load_labels(data_dir, ds)
        all_features.append(feat)
        all_labels.append(lab)
        print(f"    Features: {len(feat)} rows, Labels: {len(lab)} rows")

    features_df = pd.concat(all_features, ignore_index=True)
    labels_df = pd.concat(all_labels, ignore_index=True)

    # Impute NaN features before windowing
    meta_cols = {"pid", "date", "dataset"}
    feature_cols_in_df = [c for c in features_df.columns if c not in meta_cols]
    features_df[feature_cols_in_df] = features_df[feature_cols_in_df].fillna(
        features_df[feature_cols_in_df].median()
    )

    print(f"  Total features: {len(features_df)} rows, {len(feature_cols_in_df)} feature columns")
    print(f"  Total labels: {len(labels_df)} rows")

    # Construct windows
    print("  Constructing 28-day windows...")
    X, y, pids, feature_cols = construct_windows(features_df, labels_df)
    N_FEATURES = X.shape[2]
    print(f"  Windows: {X.shape[0]} samples, shape {X.shape}")
    print(f"  Class balance: {y.mean():.1%} positive ({y.sum()}/{len(y)})")

    # Split by user
    (X_train, y_train), (X_test, y_test) = split_by_user(X, y, pids)
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Normalize using train stats
    mean, std = compute_norm_stats(X_train)
    X_train = normalize(X_train, mean, std)
    X_test = normalize(X_test, mean, std)

    # Save
    torch.save(torch.from_numpy(X_train), paths["train_X"])
    torch.save(torch.from_numpy(y_train), paths["train_y"])
    torch.save(torch.from_numpy(X_test), paths["test_X"])
    torch.save(torch.from_numpy(y_test), paths["test_y"])
    torch.save(torch.from_numpy(mean), paths["norm_mean"])
    torch.save(torch.from_numpy(std), paths["norm_std"])
    torch.save({"n_features": N_FEATURES, "feature_cols": feature_cols}, paths["meta"])

    print(f"  Cached to {CACHE_DIR}")
    print(f"  n_features = {N_FEATURES}")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def _load_cached_split(split):
    """Load cached tensors for a split."""
    paths = get_cache_paths()
    if split == "train":
        X = torch.load(paths["train_X"], weights_only=True)
        y = torch.load(paths["train_y"], weights_only=True)
    else:
        X = torch.load(paths["test_X"], weights_only=True)
        y = torch.load(paths["test_y"], weights_only=True)
    return X, y


def get_n_features():
    """Get the number of features per day from cached metadata."""
    global N_FEATURES
    if N_FEATURES is not None:
        return N_FEATURES
    paths = get_cache_paths()
    meta = torch.load(paths["meta"], weights_only=True)
    N_FEATURES = int(meta["n_features"])
    return N_FEATURES


def get_class_weights(device="cuda"):
    """Compute inverse-frequency class weights from training labels."""
    paths = get_cache_paths()
    y = torch.load(paths["train_y"], weights_only=True)
    counts = torch.bincount(y, minlength=NUM_CLASSES).float()
    weights = counts.sum() / (NUM_CLASSES * counts)
    return weights.to(device)


def make_dataloader(batch_size, split):
    """
    Infinite dataloader for GLOBEM data.

    Yields (x, y, epoch):
        x: (B, WINDOW_DAYS, N_FEATURES) float32 on GPU
        y: (B,) long on GPU
        epoch: int (current epoch number)

    Shuffles for train, sequential for test.
    """
    assert split in ["train", "test"]
    X, y = _load_cached_split(split)
    n = len(X)
    device = torch.device("cuda")

    epoch = 1
    while True:
        if split == "train":
            perm = torch.randperm(n)
        else:
            perm = torch.arange(n)

        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = X[idx].to(device, non_blocking=True)
            y_batch = y[idx].to(device, non_blocking=True)
            yield x_batch, y_batch, epoch

        epoch += 1


def evaluate_model(model, batch_size):
    """
    Evaluate model on test set.

    Returns dict with:
        balanced_acc: balanced accuracy (primary metric)
        roc_auc: area under ROC curve (secondary metric)

    Implemented in pure PyTorch (no sklearn).
    """
    model.eval()
    X, y = _load_cached_split("test")
    device = torch.device("cuda")
    n = len(X)

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            x_batch = X[i:end].to(device)
            y_batch = y[i:end]

            logits = model(x_batch)
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]  # P(positive)
            preds = logits.argmax(dim=-1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y_batch)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    # Balanced accuracy: mean of per-class recall
    balanced_acc = _balanced_accuracy(preds, labels)

    # ROC-AUC (trapezoidal rule)
    roc_auc = _roc_auc(probs, labels)

    return {"balanced_acc": balanced_acc, "roc_auc": roc_auc}


def _balanced_accuracy(preds, labels):
    """Balanced accuracy = mean of per-class recall."""
    accs = []
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == c).float().mean().item())
    return sum(accs) / len(accs) if accs else 0.0


def _roc_auc(probs, labels):
    """ROC-AUC via trapezoidal rule. Returns 0.5 if single class."""
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by decreasing probability
    sorted_idx = torch.argsort(probs, descending=True)
    sorted_labels = labels[sorted_idx].float()

    # Compute TPR and FPR at each threshold
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    tpr = tp / n_pos
    fpr = fp / n_neg

    # Prepend (0, 0)
    tpr = torch.cat([torch.zeros(1), tpr])
    fpr = torch.cat([torch.zeros(1), fpr])

    # Trapezoidal rule
    auc = torch.trapezoid(tpr, fpr).item()
    return auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GLOBEM data for autoresearch")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to GLOBEM data directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Data directory: {data_dir}")
    print(f"Cache directory: {CACHE_DIR}")
    print()

    load_and_cache(data_dir)
    print()
    print("Done! Ready to train.")
