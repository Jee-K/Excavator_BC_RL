import argparse
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:
    raise RuntimeError(
        "This trainer requires PyTorch to be installed in the target environment."
    ) from e


# -----------------------------
# Dataset utilities
# -----------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Normalizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1.0e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def encode(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def decode(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


@dataclass
class DatasetBundle:
    obs: np.ndarray
    act: np.ndarray
    episode_id: np.ndarray
    primitive_id: np.ndarray
    phase: np.ndarray
    done: np.ndarray
    obs_keys: List[str]
    act_keys: List[str]
    primitive_names: Dict[int, str]


class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        bundle: DatasetBundle,
        context: int,
        horizon: int,
        index_tuples: List[Tuple[int, int]],
    ) -> None:
        self.bundle = bundle
        self.context = int(context)
        self.horizon = int(horizon)
        self.index_tuples = index_tuples
        self.obs_dim = int(bundle.obs.shape[1])
        self.act_dim = int(bundle.act.shape[1])

    def __len__(self) -> int:
        return len(self.index_tuples)

    def __getitem__(self, idx: int):
        ep_start, t = self.index_tuples[idx]
        ep = self.bundle.episode_id[t]
        start = max(ep_start, t - self.context + 1)
        obs_seq = np.zeros((self.context, self.obs_dim), dtype=np.float32)
        phase_seq = np.zeros((self.context, 1), dtype=np.float32)
        valid_seq = np.zeros((self.context,), dtype=np.float32)

        actual = self.bundle.obs[start : t + 1]
        actual_phase = self.bundle.phase[start : t + 1, None]
        length = actual.shape[0]
        obs_seq[-length:] = actual
        phase_seq[-length:] = actual_phase
        valid_seq[-length:] = 1.0

        act_seq = np.zeros((self.horizon, self.act_dim), dtype=np.float32)
        act_mask = np.zeros((self.horizon,), dtype=np.float32)
        for h in range(self.horizon):
            j = t + h
            if j >= len(self.bundle.act):
                break
            if self.bundle.episode_id[j] != ep:
                break
            act_seq[h] = self.bundle.act[j]
            act_mask[h] = 1.0
            if self.bundle.done[j]:
                break

        primitive_id = int(self.bundle.primitive_id[t])
        return {
            "obs_seq": torch.from_numpy(obs_seq),
            "phase_seq": torch.from_numpy(phase_seq),
            "valid_seq": torch.from_numpy(valid_seq),
            "primitive_id": torch.tensor(primitive_id, dtype=torch.long),
            "act_seq": torch.from_numpy(act_seq),
            "act_mask": torch.from_numpy(act_mask),
        }


# -----------------------------
# Model definitions
# -----------------------------


class PrimitiveConditioner(nn.Module):
    def __init__(self, num_primitives: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_primitives, emb_dim)

    def forward(self, primitive_id: torch.Tensor, steps: int) -> torch.Tensor:
        e = self.emb(primitive_id)
        return e[:, None, :].expand(-1, steps, -1)


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, num_primitives: int, emb_dim: int, hidden: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        in_dim = obs_dim + 1 + emb_dim
        self.cond = PrimitiveConditioner(num_primitives, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * act_dim),
        )
        self.act_dim = act_dim

    def forward(self, obs_seq, phase_seq, valid_seq, primitive_id):
        x = obs_seq[:, -1, :]
        p = phase_seq[:, -1, :]
        e = self.cond(primitive_id, 1)[:, 0, :]
        y = self.net(torch.cat([x, p, e], dim=-1))
        return y.view(x.shape[0], self.horizon, self.act_dim)


class GRUPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, num_primitives: int, emb_dim: int, hidden: int, horizon: int, layers: int = 2):
        super().__init__()
        self.horizon = horizon
        self.act_dim = act_dim
        self.cond = PrimitiveConditioner(num_primitives, emb_dim)
        self.gru = nn.GRU(obs_dim + 1 + emb_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * act_dim),
        )

    def forward(self, obs_seq, phase_seq, valid_seq, primitive_id):
        e = self.cond(primitive_id, obs_seq.shape[1])
        x = torch.cat([obs_seq, phase_seq, e], dim=-1)
        lengths = valid_seq.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h = h_n[-1]
        y = self.head(h)
        return y.view(x.shape[0], self.horizon, self.act_dim)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int, dropout: float):
        super().__init__()
        pad = (k - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=dilation)
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.dropout = nn.Dropout(dropout)

    def _causal(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self._causal(self.conv1, x))
        y = self.dropout(y)
        y = F.relu(self._causal(self.conv2, y))
        y = self.dropout(y)
        r = x if self.down is None else self.down(x)
        return F.relu(y + r)


class TCNPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, num_primitives: int, emb_dim: int, hidden: int, horizon: int, levels: int = 4, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.act_dim = act_dim
        self.cond = PrimitiveConditioner(num_primitives, emb_dim)
        in_ch = obs_dim + 1 + emb_dim
        blocks = []
        ch = in_ch
        for i in range(levels):
            blocks.append(TemporalBlock(ch, hidden, kernel_size, dilation=2 ** i, dropout=dropout))
            ch = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * act_dim),
        )

    def forward(self, obs_seq, phase_seq, valid_seq, primitive_id):
        e = self.cond(primitive_id, obs_seq.shape[1])
        x = torch.cat([obs_seq, phase_seq, e], dim=-1)
        x = x.transpose(1, 2)  # [B, C, T]
        h = self.tcn(x)[:, :, -1]
        y = self.head(h)
        return y.view(obs_seq.shape[0], self.horizon, self.act_dim)


def build_model(args, obs_dim: int, act_dim: int, num_primitives: int) -> nn.Module:
    if args.arch == "mlp":
        return MLPPolicy(obs_dim, act_dim, num_primitives, args.primitive_emb_dim, args.hidden_dim, args.horizon)
    if args.arch == "gru":
        return GRUPolicy(obs_dim, act_dim, num_primitives, args.primitive_emb_dim, args.hidden_dim, args.horizon, args.num_layers)
    if args.arch == "tcn":
        return TCNPolicy(obs_dim, act_dim, num_primitives, args.primitive_emb_dim, args.hidden_dim, args.horizon, args.tcn_levels, args.tcn_kernel_size, args.dropout)
    raise ValueError(f"Unknown architecture: {args.arch}")


# -----------------------------
# IO / schema
# -----------------------------


def load_npz_bundle(path: str) -> DatasetBundle:
    data = np.load(path, allow_pickle=True)

    # Accept both "obs"/"act" and "states"/"actions" key names
    if "obs" in data:
        obs = np.asarray(data["obs"], dtype=np.float32)
    elif "states" in data:
        obs = np.asarray(data["states"], dtype=np.float32)
    else:
        raise KeyError("Dataset missing required key: need 'obs' or 'states'")

    if "act" in data:
        act = np.asarray(data["act"], dtype=np.float32)
    elif "actions" in data:
        act = np.asarray(data["actions"], dtype=np.float32)
    else:
        raise KeyError("Dataset missing required key: need 'act' or 'actions'")
    n = obs.shape[0]
    if act.shape[0] != n:
        raise ValueError("obs and act must have same first dimension")

    def get_or_default(name: str, default):
        if name in data:
            arr = np.asarray(data[name])
            if arr.shape[0] != n:
                raise ValueError(f"{name} must have length {n}")
            return arr
        return default

    episode_id = get_or_default("episode_id", np.zeros((n,), dtype=np.int64)).astype(np.int64)
    primitive_id = get_or_default("primitive_id", np.zeros((n,), dtype=np.int64)).astype(np.int64)
    done = get_or_default("done", np.zeros((n,), dtype=np.bool_)).astype(np.bool_)
    phase = get_or_default("phase", np.zeros((n,), dtype=np.float32)).astype(np.float32)

    obs_keys = list(map(str, data["obs_keys"].tolist())) if "obs_keys" in data else [f"obs_{i}" for i in range(obs.shape[1])]
    act_keys = list(map(str, data["act_keys"].tolist())) if "act_keys" in data else [f"act_{i}" for i in range(act.shape[1])]

    primitive_names: Dict[int, str] = {}
    if "primitive_names_json" in data:
        primitive_names = {int(k): str(v) for k, v in json.loads(str(data["primitive_names_json"].item())).items()}
    elif "primitive_names" in data:
        raw = data["primitive_names"].tolist()
        if isinstance(raw, dict):
            primitive_names = {int(k): str(v) for k, v in raw.items()}
    else:
        primitive_names = {int(pid): f"primitive_{int(pid)}" for pid in np.unique(primitive_id)}

    return DatasetBundle(
        obs=obs,
        act=act,
        episode_id=episode_id,
        primitive_id=primitive_id,
        phase=phase,
        done=done,
        obs_keys=obs_keys,
        act_keys=act_keys,
        primitive_names=primitive_names,
    )


PRIMITIVE_NAME_MAP = {0: "dig", 1: "dump", 2: "swing", 3: "return"}


def load_directory_bundle(parent_dir: str) -> DatasetBundle:
    """Load a dataset from a directory with subfolders named 0_*, 1_*, 2_*, 3_*.

    Each subfolder's leading digit is the primitive_id. All .npz files inside
    are separate episodes. State-action pairs are built from the 'states'/'actions'
    (or 'obs'/'act') arrays in each file.
    """
    parent = Path(parent_dir)
    subfolders = sorted([d for d in parent.iterdir() if d.is_dir()])

    all_obs, all_act, all_episode_id, all_primitive_id, all_done = [], [], [], [], []
    episode_counter = 0
    primitive_names: Dict[int, str] = {}
    obs_keys: List[str] = []
    act_keys: List[str] = []

    for folder in subfolders:
        # Extract primitive id from leading digit of folder name (e.g. "0_dig_BC_data" -> 0)
        folder_name = folder.name
        if not folder_name[0].isdigit():
            continue
        prim_id = int(folder_name.split("_")[0])
        if prim_id not in primitive_names:
            primitive_names[prim_id] = PRIMITIVE_NAME_MAP.get(prim_id, f"primitive_{prim_id}")

        npz_files = sorted(glob.glob(str(folder / "*.npz")))
        if not npz_files:
            print(f"  Warning: no .npz files in {folder}")
            continue

        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)

            # Read obs/states
            if "obs" in data:
                obs = np.asarray(data["obs"], dtype=np.float32)
            elif "states" in data:
                obs = np.asarray(data["states"], dtype=np.float32)
            else:
                print(f"  Warning: skipping {npz_path} (no 'obs' or 'states' key)")
                continue

            # Read act/actions
            if "act" in data:
                act = np.asarray(data["act"], dtype=np.float32)
            elif "actions" in data:
                act = np.asarray(data["actions"], dtype=np.float32)
            else:
                print(f"  Warning: skipping {npz_path} (no 'act' or 'actions' key)")
                continue

            n = obs.shape[0]
            if act.shape[0] != n:
                print(f"  Warning: skipping {npz_path} (obs/act length mismatch)")
                continue

            all_obs.append(obs)
            all_act.append(act)
            all_episode_id.append(np.full(n, episode_counter, dtype=np.int64))
            all_primitive_id.append(np.full(n, prim_id, dtype=np.int64))

            done = np.zeros(n, dtype=np.bool_)
            done[-1] = True
            all_done.append(done)

            if not obs_keys:
                if "obs_keys" in data:
                    obs_keys = list(map(str, data["obs_keys"].tolist()))
                elif "joint_names" in data:
                    obs_keys = list(map(str, data["joint_names"].tolist()))
                else:
                    obs_keys = [f"obs_{i}" for i in range(obs.shape[1])]
                act_keys = obs_keys.copy()

            print(f"  Loaded {os.path.basename(npz_path)}: {n} samples, "
                  f"episode={episode_counter}, primitive={prim_id} ({primitive_names[prim_id]})")
            episode_counter += 1

    if not all_obs:
        raise ValueError(f"No valid .npz files found in {parent_dir}")

    obs_cat = np.concatenate(all_obs)
    act_cat = np.concatenate(all_act)
    n_total = obs_cat.shape[0]

    print(f"\nCombined dataset: {n_total} samples, {episode_counter} episodes, "
          f"{len(primitive_names)} primitives {primitive_names}")

    return DatasetBundle(
        obs=obs_cat,
        act=act_cat,
        episode_id=np.concatenate(all_episode_id),
        primitive_id=np.concatenate(all_primitive_id),
        phase=np.zeros(n_total, dtype=np.float32),
        done=np.concatenate(all_done),
        obs_keys=obs_keys,
        act_keys=act_keys,
        primitive_names=primitive_names,
    )


# -----------------------------
# Training helpers
# -----------------------------


def build_episode_split(bundle: DatasetBundle, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    episode_ids = np.unique(bundle.episode_id)
    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    val_n = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_eps = set(shuffled[:val_n].tolist())
    train_eps = [int(e) for e in shuffled if int(e) not in val_eps]
    val_eps_l = [int(e) for e in shuffled if int(e) in val_eps]
    if not train_eps:
        train_eps = val_eps_l
        val_eps_l = []
    return train_eps, val_eps_l


def build_indices(bundle: DatasetBundle, allowed_episodes: List[int]) -> List[Tuple[int, int]]:
    allowed = set(allowed_episodes)
    idxs: List[Tuple[int, int]] = []
    episode_id = bundle.episode_id
    n = len(episode_id)
    start = 0
    while start < n:
        ep = int(episode_id[start])
        end = start + 1
        while end < n and int(episode_id[end]) == ep:
            end += 1
        if ep in allowed:
            for t in range(start, end):
                idxs.append((start, t))
        start = end
    return idxs


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0.0
    for batch in loader:
        obs_seq = batch["obs_seq"].to(device)
        phase_seq = batch["phase_seq"].to(device)
        valid_seq = batch["valid_seq"].to(device)
        primitive_id = batch["primitive_id"].to(device)
        act_seq = batch["act_seq"].to(device)
        act_mask = batch["act_mask"].to(device)
        pred = model(obs_seq, phase_seq, valid_seq, primitive_id)
        mask = act_mask[..., None]
        se = ((pred - act_seq) ** 2) * mask
        ae = (pred - act_seq).abs() * mask
        denom = mask.sum().item() * act_seq.shape[-1]
        if denom <= 0:
            continue
        total_loss += se.sum().item()
        total_mae += ae.sum().item()
        total_count += denom
    if total_count == 0:
        return {"mse": float("nan"), "mae": float("nan")}
    return {"mse": total_loss / total_count, "mae": total_mae / total_count}


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if os.path.isdir(args.dataset):
        bundle_raw = load_directory_bundle(args.dataset)
    else:
        bundle_raw = load_npz_bundle(args.dataset)
    train_eps, val_eps = build_episode_split(bundle_raw, args.val_fraction, args.seed)

    train_mask = np.isin(bundle_raw.episode_id, np.asarray(train_eps))
    obs_norm = Normalizer.fit(bundle_raw.obs[train_mask])
    act_norm = Normalizer.fit(bundle_raw.act[train_mask])

    bundle = DatasetBundle(
        obs=obs_norm.encode(bundle_raw.obs),
        act=act_norm.encode(bundle_raw.act),
        episode_id=bundle_raw.episode_id,
        primitive_id=bundle_raw.primitive_id,
        phase=bundle_raw.phase,
        done=bundle_raw.done,
        obs_keys=bundle_raw.obs_keys,
        act_keys=bundle_raw.act_keys,
        primitive_names=bundle_raw.primitive_names,
    )

    train_ds = SequenceWindowDataset(bundle, args.context, args.horizon, build_indices(bundle, train_eps))
    val_ds = SequenceWindowDataset(bundle, args.context, args.horizon, build_indices(bundle, val_eps if val_eps else train_eps))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_primitives = max(bundle.primitive_names.keys()) + 1 if bundle.primitive_names else int(bundle.primitive_id.max()) + 1
    model = build_model(args, train_ds.obs_dim, train_ds.act_dim, num_primitives).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"epoch": -1, "mse": float("inf")}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            obs_seq = batch["obs_seq"].to(device)
            phase_seq = batch["phase_seq"].to(device)
            valid_seq = batch["valid_seq"].to(device)
            primitive_id = batch["primitive_id"].to(device)
            act_seq = batch["act_seq"].to(device)
            act_mask = batch["act_mask"].to(device)

            pred = model(obs_seq, phase_seq, valid_seq, primitive_id)
            mask = act_mask[..., None]
            loss = (((pred - act_seq) ** 2) * mask).sum() / (mask.sum() * act_seq.shape[-1] + 1.0e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item()
            count += 1

        train_stats = evaluate(model, train_loader, device)
        val_stats = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:04d} | train_mse={train_stats['mse']:.6f} train_mae={train_stats['mae']:.6f} "
            f"| val_mse={val_stats['mse']:.6f} val_mae={val_stats['mae']:.6f}"
        )

        if val_stats["mse"] < best["mse"]:
            best = {"epoch": epoch, "mse": float(val_stats["mse"])}
            ckpt = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "obs_dim": train_ds.obs_dim,
                "act_dim": train_ds.act_dim,
                "num_primitives": num_primitives,
                "obs_keys": bundle.obs_keys,
                "act_keys": bundle.act_keys,
                "primitive_names": bundle.primitive_names,
                "obs_norm_mean": obs_norm.mean,
                "obs_norm_std": obs_norm.std,
                "act_norm_mean": act_norm.mean,
                "act_norm_std": act_norm.std,
            }
            torch.save(ckpt, output_dir / "best_model.pt")

    manifest = {
        "dataset": os.path.abspath(args.dataset),
        "architecture": args.arch,
        "train_episodes": train_eps,
        "val_episodes": val_eps,
        "obs_keys": bundle.obs_keys,
        "act_keys": bundle.act_keys,
        "primitive_names": bundle.primitive_names,
        "best": best,
        "config": vars(args),
    }
    with open(output_dir / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved best checkpoint to {output_dir / 'best_model.pt'}")
    print(f"Saved manifest to {output_dir / 'training_manifest.json'}")


EXAMPLE_SCHEMA = {
    "obs": "float32 array [N, D_obs]",
    "act": "float32 array [N, D_act]",
    "episode_id": "int64 array [N]  -- required for proper temporal split; default all-zero",
    "primitive_id": "int64 array [N]  -- e.g. 0=dig, 1=carry, 2=dump; default all-zero",
    "phase": "float32 array [N]  -- normalized within-primitive phase in [0,1]; optional but recommended",
    "done": "bool array [N]  -- true on terminal sample of episode; optional",
    "obs_keys": "string array [D_obs] naming observation channels",
    "act_keys": "string array [D_act] naming action channels",
    "primitive_names_json": 'JSON object string, e.g. {"0": "dig", "1": "carry"}',
}


def write_schema(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(EXAMPLE_SCHEMA, f, indent=2)
    print(f"Wrote schema template to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Primitive-conditioned behavioral cloning trainer for excavator-like continuous control.")
    parser.add_argument("--dataset", type=str, help="Path to .npz file or parent directory with 0_*/1_*/2_*/3_* subfolders")
    parser.add_argument("--output-dir", type=str, default="./bc_runs/default")
    parser.add_argument("--arch", type=str, default="tcn", choices=["mlp", "gru", "tcn"])
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--primitive-emb-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tcn-levels", type=int, default=4)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--write-schema", type=str, default="", help="Write a JSON schema template and exit")
    args = parser.parse_args()

    if args.write_schema:
        write_schema(args.write_schema)
    else:
        if not args.dataset:
            parser.error("--dataset is required unless --write-schema is used")
        train(args)
