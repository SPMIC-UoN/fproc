#!/usr/bin/env python3
"""Collect AFIRM T2w data from subject folders.

For each subject subfolder in the input directory, this script copies the
specified folders and files into ./afirm_t2w_data/<subject>/.
"""

from __future__ import annotations

import argparse
import subprocess
import shutil
from pathlib import Path


FOLDERS_TO_COPY = [
]

FILES_TO_COPY = [
	"fsort/t2w/t2w.nii.gz",
	"fproc/seg_kidney_pelvis_fat_trace/kidney_pelvis_*_fat.nii.gz",
	"fproc/seg_kidney_pelvis_trace/kidney_pelvis.nii.gz",
]

FILES_TO_RESAMPLE_TO_T2W = [
	"fproc/dixon_cor_best/fat.nii.gz",
	"fproc/ff_dixon_cor/fat_fraction.nii.gz",
]


def _copy_folder(src: Path, dst: Path) -> bool:
	"""Copy folder if it exists; return True on success."""
	if not src.is_dir():
		return False
	shutil.copytree(src, dst, dirs_exist_ok=True)
	return True


def _copy_file(src: Path, dst: Path) -> bool:
	"""Copy source file to destination path."""
	if not src.is_file():
		return False

	dst.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src, dst)
	return True


def _copy_glob_matches(src_dir: Path, pattern: str, dst_dir: Path) -> int:
	"""Copy all files matching a glob pattern, keeping their filenames."""
	count = 0
	for src_file in sorted(src_dir.glob(pattern)):
		if not src_file.is_file():
			continue
		dst_name = f"{src_file.parent.name}_{src_file.name}"
		if _copy_file(src_file, dst_dir / dst_name):
			count += 1
	return count


def _resample_to_reference(src: Path, ref: Path, dst: Path) -> bool:
	"""Resample source image into reference space with FSL FLIRT."""
	if not src.is_file() or not ref.is_file():
		return False

	dst.parent.mkdir(parents=True, exist_ok=True)
	cmd = [
		"flirt",
		"-in",
		str(src),
		"-ref",
		str(ref),
		"-out",
		str(dst),
		"-applyxfm",
		"-usesqform",
		"-interp",
		"trilinear",
	]
	try:
		subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
	except FileNotFoundError as exc:
		raise RuntimeError("FSL command 'flirt' not found; cannot resample fat maps to t2w space") from exc
	except subprocess.CalledProcessError as exc:
		err = exc.stderr.strip() if exc.stderr else "Unknown FLIRT error"
		raise RuntimeError(f"FLIRT failed for {src} -> {dst}: {err}") from exc
	return True


def collect_subject_data(input_dir: Path, output_dir: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	subjects = sorted(p for p in input_dir.iterdir() if p.is_dir())
	if not subjects:
		raise ValueError(f"No subject subfolders found in {input_dir}")

	for subject_dir in subjects:
		subject = subject_dir.name
		subject_out = output_dir / subject
		subject_out.mkdir(parents=True, exist_ok=True)
		print(f"Processing {subject}")

		t2w_ref = subject_dir / "fsort/t2w/t2w.nii.gz"
		if not t2w_ref.is_file():
			print(f"  Missing t2w reference image: {t2w_ref}")

		for folder_name in FOLDERS_TO_COPY:
			src_folder = subject_dir / folder_name
			dst_folder = subject_out / Path(folder_name).name
			if _copy_folder(src_folder, dst_folder):
				print(f"  Copied folder: {folder_name}")
			else:
				print(f"  Missing folder: {src_folder}")

		for rel_src in FILES_TO_COPY:
			matched = _copy_glob_matches(subject_dir, rel_src, subject_out)
			if matched:
				print(f"  Copied {matched} file(s) from pattern: {rel_src}")
			else:
				print(f"  No matches for pattern: {subject_dir / rel_src}")

		for rel_src in FILES_TO_RESAMPLE_TO_T2W:
			for src_file in sorted(subject_dir.glob(rel_src)):
				if not src_file.is_file():
					continue
				dst_name = f"{src_file.parent.name}_{src_file.name}"
				dst_file = subject_out / dst_name
				if _resample_to_reference(src_file, t2w_ref, dst_file):
					print(f"  Resampled to t2w space: {rel_src}")
				else:
					print(f"  Could not resample (missing input/ref): {src_file}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Copy specified AFIRM T2w files/folders from each subject subfolder "
			"in input directory to ./afirm_t2w_data"
		)
	)
	parser.add_argument("input_dir", help="Input directory containing subject subfolders")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_dir = Path(args.input_dir).resolve()
	if not input_dir.is_dir():
		raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

	output_dir = Path("/spmstore/project/RenalMRI/afirm/afirm_t2w_data")
	collect_subject_data(input_dir, output_dir)
	print(f"Done. Output written to {output_dir}")


if __name__ == "__main__":
	main()
