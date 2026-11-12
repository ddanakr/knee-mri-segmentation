import os
import glob
import random
from collections import OrderedDict
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler



def gather_pairs(images_dir: str, labels_dir: str, image_suffix: str = "_0000"):

    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    pairs = []

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        case_id = fname.replace(image_suffix + ".nii.gz", "").replace(".nii.gz", "")
        mask_path = os.path.join(labels_dir, f"{case_id}.nii.gz")

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"Mask not found for {fname}, skipping")

    print(f"Found {len(pairs)} pairs of image/mask")
    return pairs

def gather_processed_pairs(processed_dir: str):

    image_paths = sorted(glob.glob(os.path.join(processed_dir, "*_image.npy")))
    pairs = []

    for img_path in image_paths:
        case_id = os.path.basename(img_path)[: -len("_image.npy")]
        mask_path = os.path.join(processed_dir, f"{case_id}_mask.npy")

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"Mask not found for {case_id}, skipping")

    print(f"Found {len(pairs)} processed pairs of image/mask")
    return pairs

def _load_shape(path: str):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r").shape
    return nib.load(path).shape

def _load_array(path: str):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return nib.load(path).get_fdata()

def slice_along_axis(arr, idx: int, axis: int):
    """Basic-indexing slice (fast view on memmaps), unlike np.take which always copies via a
    generic gather loop and is pathologically slow on memmap-backed arrays."""
    indexer = [slice(None)] * arr.ndim
    indexer[axis] = idx
    return arr[tuple(indexer)]

def build_slice_index(pairs, axis: int = 2, min_foreground_voxels: int = 0):
    slice_index = []

    for vol_idx, (img_path, mask_path) in enumerate(pairs):
        shape = _load_shape(img_path)
        n_slices = shape[axis]

        if min_foreground_voxels > 0:
            mask_data = _load_array(mask_path)
            for s in range(n_slices):
                sl = slice_along_axis(mask_data, s, axis)
                if np.count_nonzero(sl) >= min_foreground_voxels:
                    slice_index.append((vol_idx, s))
        else:
            for s in range(n_slices):
                slice_index.append((vol_idx, s))

    return slice_index

def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std() + 1e-8
    volume = (volume - mean) / std
    return volume


class KneeDataset(Dataset):
    def __init__(self, pairs, slice_index, axis=2, target_size=(256, 256), transform=None, cache_size=2):
        self.pairs = pairs
        self.slice_index = slice_index
        self.axis = axis
        self.target_size = target_size
        self.transform = transform
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.slice_index)

    def _load_volume(self, vol_idx):
        if vol_idx in self._cache:
            self._cache.move_to_end(vol_idx)
            return self._cache[vol_idx]

        img_path, mask_path = self.pairs[vol_idx]

        if img_path.endswith(".npy"):
            image = np.load(img_path, mmap_mode="r")
            mask = np.load(mask_path, mmap_mode="r")
        else:
            image = nib.load(img_path).get_fdata().astype(np.float32)
            image = normalize_intensity(image)
            mask = nib.load(mask_path).get_fdata().astype(np.int64)

        self._cache[vol_idx] = (image, mask)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return image, mask

    def __getitem__(self, idx):
        vol_idx, slice_idx = self.slice_index[idx]
        image, mask = self._load_volume(vol_idx)

        img_slice = np.ascontiguousarray(slice_along_axis(image, slice_idx, self.axis))
        mask_slice = np.ascontiguousarray(slice_along_axis(mask, slice_idx, self.axis))

        img_tensor = torch.from_numpy(img_slice).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_slice).unsqueeze(0).long()

        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).float(), size=self.target_size, mode="nearest"
        ).squeeze(0).long()

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor


class ChunkedVolumeSampler(Sampler):
    """Shuffles volume order each epoch, and shuffles slices within a chunk of volumes at a
    time, instead of a pure global shuffle across all slices.

    The full processed dataset doesn't fit in RAM (~40GB of .npy vs ~13GB Colab RAM), so a
    global shuffle makes almost every sample a cold random-access read of a different large
    file (~800ms/sample measured), instead of a warm cached read (~19ms/sample). Keeping a
    small chunk of volumes "hot" at a time and draining all their slices before moving on
    gets most of the warm-read speed back while still reshuffling every epoch.
    """

    def __init__(self, slice_index, chunk_size: int = 32, seed: int = 0):
        self.chunk_size = chunk_size
        self.seed = seed
        self.epoch = 0

        self.vol_to_indices = {}
        for i, (vol_idx, _) in enumerate(slice_index):
            self.vol_to_indices.setdefault(vol_idx, []).append(i)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        vol_ids = list(self.vol_to_indices.keys())
        rng.shuffle(vol_ids)

        for start in range(0, len(vol_ids), self.chunk_size):
            chunk_indices = []
            for vol_idx in vol_ids[start:start + self.chunk_size]:
                chunk_indices.extend(self.vol_to_indices[vol_idx])
            rng.shuffle(chunk_indices)
            yield from chunk_indices

    def __len__(self):
        return sum(len(v) for v in self.vol_to_indices.values())


if __name__ == "__main__":
    images_dir = "/content/data/raw/imagesTr"
    labels_dir = "/content/data/raw/labelsTr"

    pairs = gather_pairs(images_dir, labels_dir)
    slice_index = build_slice_index(pairs[:2])

    ds = KneeDataset(pairs[:2], slice_index)
    print("dataset length:", len(ds))

    img, mask = ds[0]
    print("image shape:", img.shape, "dtype:", img.dtype, "min/max:", img.min().item(), img.max().item())
    print("mask shape:", mask.shape, "dtype:", mask.dtype, "unique values:", torch.unique(mask))