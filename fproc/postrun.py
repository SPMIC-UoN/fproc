"""
FPROC: Post run command to combine output from subjects into CSV files
"""

import argparse
from datetime import datetime
import os
import subprocess
import logging
import importlib
import sys
import os
import csv
import re
from openpyxl import Workbook

LOG = logging.getLogger(__name__)


def run_fproc_combine(input_dir, output_file, paths, subjids=None, allow_text=False):
    """
    Run fproc-combine command

    Args:
        input_dir: Input directory path
        output_file: Output CSV file path
        paths: List of paths to combine
        subjids: Optional subject IDs file path
    """
    cmd = [
        "fproc-combine",
        "--input",
        input_dir,
        "--output",
        output_file,
    ]

    if subjids:
        cmd.extend(["--subjids", subjids])

    cmd.append("--path")
    cmd.extend(paths)

    if allow_text:
        cmd.append("--allow-text")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def csv_to_xlsx(csv_path: str, xlsx_path: str) -> None:
    """
    Convert a CSV file to XLSX, freeze top row and first column, and auto-size columns.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "results"

    max_widths = []
    def _try_parse_number(s: str):
        if s is None:
            return None
        s2 = s.strip()
        if s2 == "":
            return None
        # Do not convert values with leading zero (likely IDs)
        if re.match(r"^0\d+", s2):
            return s
        # Remove thousand separators if present (e.g., 1,234.56)
        s3 = s2.replace(",", "")
        # Integer
        if re.match(r"^[+-]?\d+$", s3):
            try:
                return int(s3)
            except Exception:
                return s
        # Float (including scientific)
        if re.match(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$", s3):
            try:
                return float(s3)
            except Exception:
                return s
        return s

    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        for r_idx, row in enumerate(reader, start=1):
            for c_idx, cell in enumerate(row, start=1):
                # Preserve header row as text
                if r_idx == 1:
                    value = cell
                else:
                    value = _try_parse_number(cell)
                ws.cell(row=r_idx, column=c_idx, value=value)
                text = str(cell) if cell is not None else ""
                # compute display width from original text
                needed = len(text)
                if len(max_widths) < c_idx:
                    max_widths.append(needed)
                else:
                    if needed > max_widths[c_idx - 1]:
                        max_widths[c_idx - 1] = needed

    # Freeze top row and first column - keep headers and first column visible
    ws.freeze_panes = "B2"

    # Set column widths
    for i, w in enumerate(max_widths, start=1):
        col_letter = ws.cell(row=1, column=i).column_letter
        width = min(max(w + 2, 8), 50)
        ws.column_dimensions[col_letter].width = width

    wb.save(xlsx_path)


def run_fproc_flatten(input_dir, output_dir):
    """
    Run fproc-flatten command

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
    """
    cmd = [
        "fproc-flatten",
        "--input",
        input_dir,
        "--output",
        output_dir,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-processing script - combines CSV outputs from fproc pipeline"
    )
    parser.add_argument(
        "--pipeline",
        required=True,
        help="Name of python module containing pipeline config, or path to .py file",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory - may be relative to pipeline.BASE",
    )
    parser.add_argument(
        "--flatten", action="store_true", help="Run fproc-flatten to flatten images"
    )
    parser.add_argument(
        "--xlsx",
        action="store_true",
        help="Write outputs in XLSX format instead of CSV (converts CSV to XLSX)",
    )
    return parser.parse_args()


def main():
    options = parse_args()
    dateout = datetime.now().strftime("%Y%m%d")
    LOG.info("FPROC post-processing script")
    LOG.info(f" - Date: {dateout}")

    try:
        LOG.info(f" - Loading configuration from {options.pipeline}")
        pipeline = importlib.import_module(options.pipeline)
        pipeline_config_fname = pipeline.__name__
    except ImportError:
        pipeline_config_fpath = os.path.abspath(os.path.normpath(options.pipeline))
        pipeline_config_dirname, pipeline_config_fname = os.path.split(
            pipeline_config_fpath
        )
        try:
            sys.path.append(pipeline_config_dirname)
            pipeline = importlib.import_module(pipeline_config_fname.replace(".py", ""))
        except ImportError:
            LOG.exception("Loading config")
            raise ValueError(
                f"Could not load configuration {options.pipeline} - must be a python module or file"
            )
        finally:
            sys.path.remove(pipeline_config_dirname)

    if os.path.isdir(options.input):
        input_dir = options.input
    else:
        input_dir = os.path.join(pipeline.STUDYDIR, options.input)

    # Create output directory
    input_base = os.path.basename(input_dir)
    output_base = os.path.join(pipeline.STUDYDIR, "csv", input_base)
    os.makedirs(output_base, exist_ok=True)

    LOG.info(f" - Study: {pipeline.NAME}")
    LOG.info(f" - Input directory: {input_dir}")
    LOG.info(f" - Output directory: {output_base}")

    allow_text = getattr(pipeline, "ALLOW_TEXT", [])
    for cohort_name, cohort_prefix, subjids in pipeline.COHORTS:
        LOG.info(f" - Processing {cohort_name}")

        # Process standard output files
        for file_id, paths in pipeline.OUTFILES.items():
            base_name = f"{pipeline.NAME}_{cohort_prefix}{file_id}_{dateout}"
            csv_path = os.path.join(output_base, f"{base_name}.csv")
            xlsx_path = os.path.join(output_base, f"{base_name}.xlsx")
            run_fproc_combine(
                input_dir,
                csv_path,
                paths,
                subjids=subjids,
                allow_text=file_id in allow_text,
            )
            if options.xlsx:
                try:
                    LOG.info(" - Converting CSV to XLSX: %s -> %s", csv_path, xlsx_path)
                    csv_to_xlsx(csv_path, xlsx_path)
                    # remove intermediate CSV
                    try:
                        os.remove(csv_path)
                    except Exception:
                        LOG.warning("Could not remove intermediate CSV %s", csv_path)
                except Exception:
                    LOG.exception("Failed to convert CSV to XLSX for %s", csv_path)

    # Flatten images (optional)
    if options.flatten:
        LOG.info(" - Flattening images with fproc-flatten")
        run_fproc_flatten(input_dir, os.path.join(output_base, f"imgs_{dateout}"))

    LOG.info("DONE")
