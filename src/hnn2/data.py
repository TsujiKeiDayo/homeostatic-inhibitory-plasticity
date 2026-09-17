"""Explicit MNIST preparation and reuse of one saved dataset NPZ."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.transform import resize as _skimage_resize

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Dataset:
    """Loaded arrays plus the identity of the exact input bytes; reusable in memory."""

    arrays: dict[str, np.ndarray]
    source: dict


def load_dataset(path: str | Path | Dataset) -> Dataset:
    """Read an NPZ with its checksum and metadata, or reuse a loaded Dataset."""
    if isinstance(path, Dataset):
        return path
    path = Path(path).expanduser().resolve()
    content = path.read_bytes()
    with np.load(BytesIO(content), allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files if key != "meta"}
        metadata = json.loads(str(source["meta"])) if "meta" in source.files else {}
    return Dataset(arrays, {
        "path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
        "metadata": metadata, "shapes": {k: list(v.shape) for k, v in arrays.items()}
    })


def load_raw_mnist(raw_root: Path) -> dict[str, np.ndarray]:
    """Load MNIST as float arrays in [0, 1], downloading it if absent."""
    # Lazy import: only the explicitly requested dataset build needs torchvision.
    from torchvision.datasets import MNIST

    train = MNIST(root=str(raw_root), train=True, download=True)
    test = MNIST(root=str(raw_root), train=False, download=True)
    return {
        "x_train_full": train.data.numpy().astype(np.float32) / 255.0,
        "y_train_full": np.asarray(train.targets, dtype=np.int64),
        "x_test_full": test.data.numpy().astype(np.float32) / 255.0,
        "y_test_full": np.asarray(test.targets, dtype=np.int64),
    }


def allocate_proportional_counts(labels: np.ndarray, sample_count: int) -> dict[int, int]:
    """Class-proportional counts summing to exactly ``sample_count``."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    classes, class_counts = np.unique(labels, return_counts=True)
    total = int(class_counts.sum())
    if sample_count > total:
        raise ValueError(f"requested {sample_count} samples from a pool of {total}")

    # Give the leftover slots to the classes that lost the most to flooring.
    ideal = sample_count * (class_counts / total)
    alloc = np.floor(ideal).astype(int)
    remainder = sample_count - int(alloc.sum())
    if remainder > 0:
        order = np.argsort(-(ideal - alloc))
        alloc[order[:remainder]] += 1
    return {int(c): int(n) for c, n in zip(classes, alloc)}


def stratified_sample_indices(
    labels: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample indices without replacement, preserving the pool's class proportions.

    Returns shuffled indices of shape (sample_count,) and advances ``rng``.
    """
    counts = allocate_proportional_counts(labels, sample_count)
    chosen: list[np.ndarray] = []
    for cls, n_take in counts.items():
        cls_idx = np.nonzero(labels == cls)[0]
        if n_take > 0:
            chosen.append(rng.choice(cls_idx, size=n_take, replace=False))
    idx = np.concatenate(chosen)
    # Undo the per-class grouping so consumers batching in order see mixed classes.
    rng.shuffle(idx)
    return idx


def resize_flat(images: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize images with anti-aliasing and flatten each image into one row.

    Returns float32 values of shape (n_samples, size[0] * size[1]).
    """
    out = np.stack(
        [_skimage_resize(img, size, anti_aliasing=True) for img in images]
    ).astype(np.float32)
    return out.reshape(out.shape[0], -1)


@dataclass(frozen=True)
class DatasetParams:
    """Dataset recipe saved alongside the arrays in the user-named NPZ."""

    image_size: tuple[int, int]
    raw_baseline_size: tuple[int, int]
    n_samples_per_split: int
    train_val_ratio: float
    data_seed: int

    @classmethod
    def from_config(cls, config) -> "DatasetParams":
        """Extract a recipe with normalised types for consistent content hashing."""
        return cls(
            image_size=tuple(config.image_size),
            raw_baseline_size=tuple(config.raw_baseline_size),
            n_samples_per_split=int(config.n_samples_per_split),
            train_val_ratio=float(config.train_val_ratio),
            data_seed=int(config.data_seed),
        )

    def to_json(self) -> str:
        """Serialise the recipe in stable key order for the dataset content hash."""
        return json.dumps(
            {
                "image_size": list(self.image_size),
                "raw_baseline_size": list(self.raw_baseline_size),
                "n_samples_per_split": self.n_samples_per_split,
                "train_val_ratio": self.train_val_ratio,
                "data_seed": self.data_seed,
            },
            sort_keys=True,
        )


def build_dataset_arrays(params: DatasetParams, raw_root: Path) -> dict[str, np.ndarray]:
    """Prepare sampled splits and the full test split at both image resolutions.

    Images are float32 sample rows; labels are aligned int64 vectors. Raw MNIST
    is downloaded to ``raw_root`` if absent. See SPEC.md, section 4.
    """
    raw = load_raw_mnist(raw_root)

    # Independent sub-streams, so changing the split does not shift the sampling.
    split_rng, sampling_rng = (
        np.random.default_rng(s)
        for s in np.random.SeedSequence(params.data_seed).spawn(2)
    )

    # Split the shuffled train pool at train_val_ratio so validation does not
    # follow the original file order.
    perm = split_rng.permutation(len(raw["x_train_full"]))
    n_train = int(len(perm) * params.train_val_ratio)
    pools = {
        "train": (raw["x_train_full"][perm[:n_train]], raw["y_train_full"][perm[:n_train]]),
        "val": (raw["x_train_full"][perm[n_train:]], raw["y_train_full"][perm[n_train:]]),
        "test": (raw["x_test_full"], raw["y_test_full"]),
    }

    # Both resolutions come from the SAME sampled images, so the raw-pixel
    # control arm differs from the model input only in resolution.
    small, big = params.image_size, params.raw_baseline_size
    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        x_pool, y_pool = pools[split]
        idx = stratified_sample_indices(y_pool, params.n_samples_per_split, sampling_rng)
        images = x_pool[idx]
        arrays[f"x_{split}_small"] = resize_flat(images, small)
        arrays[f"x_{split}_big"] = resize_flat(images, big)
        arrays[f"y_{split}"] = y_pool[idx]

    # Full-test evaluation uses every test image in its original order.
    arrays["x_test_full_small"] = resize_flat(raw["x_test_full"], small)
    arrays["x_test_full_big"] = resize_flat(raw["x_test_full"], big)
    arrays["y_test_full"] = raw["y_test_full"]
    return arrays


def dataset_content_hash(arrays: dict[str, np.ndarray], params: DatasetParams) -> str:
    """Return a 12-character SHA-256 prefix over the recipe, keys and array bytes.

    The hash is independent of dict ordering.
    """
    digest = hashlib.sha256()
    digest.update(params.to_json().encode("utf-8"))
    for key in sorted(arrays):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()[:12]


def create_dataset(path: str | Path, params: DatasetParams, raw_root: str | Path) -> Path:
    """Explicit preparation only: download/cache MNIST, build once, refuse overwrite."""
    import importlib.metadata
    from .result_io import _npz

    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(path)

    arrays = build_dataset_arrays(params, Path(raw_root))
    metadata = {"dataset": "MNIST", "params": json.loads(params.to_json()),
                "preprocessing": "float32 / 255; skimage resize anti_aliasing=True; flatten",
                "raw_root": str(Path(raw_root).resolve()),
                "content_hash": dataset_content_hash(arrays, params),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scikit-image", "torchvision")
                }}

    path.parent.mkdir(parents=True, exist_ok=True)
    _npz(path, {**arrays, "meta": np.array(json.dumps(metadata, sort_keys=True))})
    return path
