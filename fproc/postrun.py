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

LOG = logging.getLogger(__name__)


def run_fproc_combine(input_dir, output_file, paths, subjids=None):
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

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


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

    for cohort_name, cohort_prefix, subjids in pipeline.COHORTS:
        LOG.info(f" - Processing {cohort_name}")

        # Process standard output files
        for file_id, paths in pipeline.OUTFILES.items():
            run_fproc_combine(
                input_dir,
                os.path.join(
                    output_base,
                    f"{pipeline.NAME}_{cohort_prefix}{file_id}_{dateout}.csv",
                ),
                paths,
                subjids=subjids,
            )

    # Flatten images (optional)
    if options.flatten:
        LOG.info(" - Flattening images with fproc-flatten")
        run_fproc_flatten(input_dir, os.path.join(output_base, f"imgs_{dateout}"))

    LOG.info("DONE")
