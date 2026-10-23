#!/usr/bin/env python3
"""
Scan subject folders and compare affines between:
 - fsort/t1_molli/t1_map.nii.gz
 - fsort/t1_molli_raw/*.nii.gz

Warn when affines are different.
"""

import argparse
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np


def _warn(msg):
    print(f"WARN: {msg}")


def _info(msg):
    print(f"INFO: {msg}")


def _compare_affines(t1_map_path, raw_path, atol, rtol):
    t1_map_img = nib.load(str(t1_map_path))
    raw_img = nib.load(str(raw_path))
    return np.allclose(t1_map_img.affine, raw_img.affine, atol=atol, rtol=rtol)


def _fix_affine(t1_map, raw_file):
    try:
        subprocess.run(
            ["fslcpgeom", str(t1_map), str(raw_file), "-d"],
            check=True,
            capture_output=True,
            text=True,
        )
        _info(f"  Fixed: copied affine from {t1_map.name} to {raw_file.name}")
    except subprocess.CalledProcessError as exc:
        _warn(f"  fslcpgeom failed for {raw_file}: {exc.stderr.strip()}")


def scan_subject(subject_dir, atol, rtol, fix=False):
    t1_map = subject_dir / "fsort" / "t1_molli" / "t1_map.nii.gz"
    raw_dir = subject_dir / "fsort" / "t1_molli_raw"
    raw_files = sorted(raw_dir.glob("*.nii.gz")) if raw_dir.exists() else []

    warned = False

    if not t1_map.exists():
        t1_map = subject_dir / "fsort" / "t1_molli" / "t1_map_1.nii.gz"
        if not t1_map.exists():
            _warn(f"{subject_dir.name}: missing {t1_map}")
            return True

    if not raw_files:
        _warn(f"{subject_dir.name}: no raw files found in {raw_dir}")
        return True

    for raw_file in raw_files:
        try:
            same = _compare_affines(t1_map, raw_file, atol=atol, rtol=rtol)
        except Exception as exc:
            _warn(f"{subject_dir.name}: failed to compare {t1_map} vs {raw_file}: {exc}")
            warned = True
            continue

        if not same:
            _warn(
                f"{subject_dir.name}: affine mismatch between {t1_map} and {raw_file}"
            )
            if fix:
                _fix_affine(t1_map, raw_file)
            warned = True

    return warned


def main():
    parser = argparse.ArgumentParser(
        description="Check affine consistency between t1_molli map and raw MOLLI NIfTI files"
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing subject subdirectories",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for affine comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for affine comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        default=False,
        help="Fix mismatches by copying affine from t1_map to each raw file using fslcpgeom",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_folder}")

    subject_dirs = sorted([p for p in input_folder.iterdir() if p.is_dir()])
    if not subject_dirs:
        _warn(f"No subject directories found under {input_folder}")
        raise SystemExit(1)

    _info(f"Scanning {len(subject_dirs)} subject directories under {input_folder}")
    any_warnings = False
    for subject_dir in subject_dirs:
        any_warnings = scan_subject(subject_dir, args.atol, args.rtol, args.fix) or any_warnings

    if any_warnings:
        _warn("Finished with warnings")
        raise SystemExit(1)

    _info("Finished: no affine mismatches found")


if __name__ == "__main__":
    main()
