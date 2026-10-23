#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np


TARGET = "fsort/t1_molli_raw/t1_molli_raw.nii.gz"


def reorder_40_volumes(data):
    """Flatten the final 5x8 block to 40 volumes, reorder them, then reshape back."""
    if data.ndim < 2:
        return None

    last_two = data.shape[-2:]
    if last_two != (5, 8):
        return None

    reshaped = data.reshape(data.shape[:-2] + (40,))
    reordered = np.empty_like(reshaped)

    for idx in range(40):
        src = reshaped[..., idx]
        dest_idx = (5 * idx) % 40 + (4 - idx // 8)
        reordered[..., dest_idx] = src

    return reordered.reshape(data.shape)


def process_subject(subject_dir: Path):
    src = subject_dir / TARGET
    if not src.is_file():
        return False

    img = nib.load(str(src))
    data = np.asarray(img.get_fdata())

    reordered = reorder_40_volumes(data)
    if reordered is None:
        print(f"Skipping {src}: unexpected shape {data.shape}")
        return False

    backup = subject_dir / "fsort/t1_molli_raw/t1_molli_raw_orig.nii.gz"
    if backup.exists():
        print(f"Skipping {src}: backup already exists at {backup}")
        return False

    # Move the original aside and save the reordered image back in-place.
    os.makedirs(backup.parent, exist_ok=True)
    os.replace(src, backup)

    final_img = nib.Nifti1Image(reordered, img.affine, img.header)
    nib.save(final_img, str(src))
    return True


def iter_man_dirs(root: Path):
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith("MAN_"):
            yield entry


def main():
    parser = argparse.ArgumentParser(
        description="Reorder 5x8 stacks in MAN_ subfolders for fsort/t1_molli_raw/t1_molli_raw.nii.gz"
    )
    parser.add_argument("folder", type=Path, help="Root folder containing MAN_ subfolders")
    args = parser.parse_args()

    root = args.folder.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder does not exist: {root}")

    count = 0
    for man_dir in iter_man_dirs(root):
        if process_subject(man_dir):
            count += 1
            print(f"Processed {man_dir / TARGET}")

    print(f"Processed {count} file(s)")


if __name__ == "__main__":
    main()
