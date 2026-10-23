#!/usr/bin/env python3
"""Collect AFIRM T1 data from subject folders.

For each subject subfolder in the input directory, this script copies the
specified folders and files into ./afirm_t1_data/<subject>/.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


FOLDERS_TO_COPY = [
	"fproc/seg_kidney_t1_mdr_clean_native_proc",
	"fproc/seg_kidney_t1_se_clean_native",
]

FILES_TO_COPY = [
	"fproc/t1_molli_stitch/*t1_map.nii.gz",
	"fsort/t1_molli_raw/t1_molli_raw*.nii.gz",
	"fproc/t1_molli_mdr_stitch/*t1_map.nii.gz",
	"fsort/t1_se_raw/t1_se_mag.nii.gz",
	"fproc/t1_se_mdr_stitch/*t1_map.nii.gz",
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Copy specified AFIRM T1 files/folders from each subject subfolder "
			"in input directory to ./afirm_t1_data"
		)
	)
	parser.add_argument("input_dir", help="Input directory containing subject subfolders")
	parser.add_argument("output_dir", help="Output folder")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_dir = Path(args.input_dir).resolve()
	if not input_dir.is_dir():
		raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

	output_dir = Path(args.output_dir).resolve()
	if not output_dir.is_dir():
                os.makedirs(output_dir)
	collect_subject_data(input_dir, output_dir)
	print(f"Done. Output written to {output_dir}")


if __name__ == "__main__":
	main()
