#!/usr/bin/env python3
"""Warn for NIfTI files with singular qform or sform matrices."""

from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def _is_singular(mat: np.ndarray) -> bool:
	"""Return True when matrix is singular or not a 4x4 affine."""
	if mat is None:
		return True
	arr = np.asarray(mat)
	if arr.shape != (4, 4):
		return True
	return np.linalg.matrix_rank(arr) < 4


def check_file(fpath: Path) -> tuple[list[str], bool, bool]:
	"""Return warnings and singularity flags for qform and sform."""
	warnings: list[str] = []
	img = nib.load(str(fpath))

	qform, _ = img.get_qform(coded=True)
	sform, _ = img.get_sform(coded=True)
	q_singular = _is_singular(qform)
	s_singular = _is_singular(sform)

	if q_singular:
		warnings.append("qform is singular")
	if s_singular:
		warnings.append("sform is singular")

	return warnings, q_singular, s_singular


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Scan NIfTI files matching a glob and warn for singular qform/sform"
	)
	parser.add_argument(
		"glob_expr",
		help="Glob expression for NIfTI files (example: 'data/**/*.nii.gz')",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	matches = sorted(glob.glob(args.glob_expr, recursive=True))
	if not matches:
		print(f"No files matched glob: {args.glob_expr}")
		return 1

	warned = 0
	fixed = 0
	for match in matches:
		fpath = Path(match)
		try:
			warnings, _, s_singular = check_file(fpath)
		except Exception as exc:
			print(f"WARNING: {fpath}: failed to read NIfTI ({exc})")
			warned += 1
			continue

		for msg in warnings:
			print(f"WARNING: {fpath}: {msg}")
			warned += 1

		if s_singular:
			result = subprocess.run(
				["fslorient", "-copyqform2sform", str(fpath)],
				check=False,
				capture_output=True,
				text=True,
			)
			if result.returncode == 0:
				fixed += 1
				print(f"FIXED: {fpath}: copied qform to sform")
			else:
				print(
					f"WARNING: {fpath}: fslorient failed (exit {result.returncode}): "
					f"{result.stderr.strip() or result.stdout.strip()}"
				)
				warned += 1

	print(f"Scanned {len(matches)} file(s). Warnings: {warned}. Fixed sform: {fixed}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
