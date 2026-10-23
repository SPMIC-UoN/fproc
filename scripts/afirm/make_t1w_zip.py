#!/usr/bin/env python3
"""Zip up AFIRM T1w images from subject folders.

For each subject subfolder in the input directory, this script finds
fsort/t1w/t1w.nii.gz and adds it to a zip file as T1_<subjid>.nii.gz.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

T1W_REL_PATH = "fsort/t1w/t1w.nii.gz"


def collect_subject_t1w(input_dir: Path, output_zip: Path) -> None:
	subjects = sorted(p for p in input_dir.iterdir() if p.is_dir())
	if not subjects:
		raise ValueError(f"No subject subfolders found in {input_dir}")

	output_zip.parent.mkdir(parents=True, exist_ok=True)
	added, missing = 0, 0
	with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
		for subject_dir in subjects:
			subjid = subject_dir.name
			t1w_file = subject_dir / T1W_REL_PATH
			if not t1w_file.is_file():
				print(f"  Missing T1w image for {subjid}: {t1w_file}")
				missing += 1
				continue

			arcname = f"T1_{subjid}.nii.gz"
			zf.write(t1w_file, arcname)
			print(f"  Added {arcname}")
			added += 1

	print(f"Done. Added {added} file(s), {missing} missing. Output written to {output_zip}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			f"Collect {T1W_REL_PATH} from each subject subfolder in input directory "
			"into a zip file, one entry per subject named T1_<subjid>.nii.gz"
		)
	)
	parser.add_argument("input_dir", help="Input directory containing subject subfolders")
	parser.add_argument(
		"-o",
		"--output",
		default="afirm_t1w_data.zip",
		help="Output zip file path (default: ./afirm_t1w_data.zip)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_dir = Path(args.input_dir).resolve()
	if not input_dir.is_dir():
		raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

	collect_subject_t1w(input_dir, Path(args.output).resolve())


if __name__ == "__main__":
	main()
