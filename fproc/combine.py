"""
FPROC: Combine stats output from previous runs
"""
from collections import OrderedDict
import glob
import logging
import os
import csv
import sys

from ._version import __version__
from .options import StatsCombineArgumentParser

LOG = logging.getLogger(__name__)

def main():
    arg_parser = StatsCombineArgumentParser()
    options = arg_parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    LOG.info(f"FPROC v{__version__} - OUTPUT COMBINE\n")
    LOG.info(f" - Input directory: {options.input}")
    LOG.info(f" - Output file: {options.output}")

    if options.subjids:
        with open(options.subjids) as f:
            subjids = [l.strip() for l in f.readlines() if os.path.isdir(os.path.join(options.input, l.strip()))]
    else:
        subjids = sorted([d for d in os.listdir(options.input) if os.path.isdir(os.path.join(options.input, d))])

    paths = options.path
    if options.paths:
        with open(options.paths) as f:
            paths += [l.strip() for l in f.readlines() if l.strip() != ""]

    stats = []
    for subjid in subjids:
        LOG.info(f" - Subject ID: {subjid}")
        subjdir = os.path.join(options.input, subjid)
        add_subj_stats(subjid, os.path.join(options.input, subjid), paths, stats)

    with open(options.output, "w") as f:
        headers = []
        for row in stats:
            for key in row:
                # Preserve order
                if key not in headers: headers.append(key)
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)

def add_stats(fglob, subj_stats):
    fnames = list(glob.glob(fglob))
    if not fnames:
        LOG.warn(f" - Failed to find any file matching {fglob}")
        return
    elif len(fnames) > 1:
        LOG.warn(f" - Multiple files matching {fglob} - will use first which is {fnames[0]}")
    fname = fnames[0]

    LOG.info(f" - Adding stats from {fname}")
    with open(fname) as f:
        for line in f.readlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                LOG.warn(f" - {fname}: Ignoring line: {line}, did not contain key, value pair")
            else:
                try:
                    subj_stats[parts[0]] = float(parts[1])
                except ValueError:
                    LOG.warn(f" - {fname}: Ignoring line: {line}, value was not numeric")

def add_subj_stats(subjid, subjdir, paths, stats):
    subj_stats = OrderedDict()
    subj_stats["subjid"] = subjid
    for rel_path in paths:
        fpath = os.path.join(subjdir, rel_path)
        add_stats(fpath, subj_stats)

    stats.append(subj_stats)
