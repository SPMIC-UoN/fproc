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
            reader = csv.reader(f)
            subjids = sorted([row[options.subjids_col].strip() for row in reader])
    else:
        subjids = sorted([d for d in os.listdir(options.input) if os.path.isdir(os.path.join(options.input, d))])

    paths = options.path
    if options.paths:
        with open(options.paths) as f:
            paths += [l.strip() for l in f.readlines() if l.strip() != ""]

    csv_paths = options.path_csv
    if options.paths_csv:
        with open(options.paths_csv) as f:
            csv_paths += [l.strip() for l in f.readlines() if l.strip() != ""]

    stats = []
    for subjid in subjids:
        LOG.info(f" - Subject ID: {subjid}")
        subjdir = os.path.join(options.input, subjid)
        if not os.path.isdir(subjdir):
            LOG.warn(f" - Subject directory {subjdir} does not exist")
            continue
        subj_stats = OrderedDict()
        subj_stats["subjid"] = subjid
        add_csv_stats(subjid, subjdir, csv_paths, subj_stats)
        add_kv_stats(subjid, subjdir, paths, subj_stats)
        stats.append(subj_stats)

    with open(options.output, "w") as f:
        headers = []
        for row in stats:
            for key in row:
                if options.skip_empty and row[key] == 0:
                    continue
                else:
                    # Preserve order
                    if key not in headers: headers.append(key)

        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in stats:
            row_pruned = {
                k : v for k, v in row.items() if k in headers
            }
            writer.writerow(row_pruned)

def add_kv_stats(subjid, subjdir, paths, subj_stats):
    for rel_path in paths:
        fglob = os.path.join(subjdir, rel_path)
        fnames = list(glob.glob(fglob))
        if not fnames:
            LOG.warn(f" - Failed to find any file matching {fglob}")
            continue
        elif len(fnames) > 1:
            LOG.warn(f" - Multiple files matching {fglob} - will use first which is {fnames[0]}")
        fname = fnames[0]

        LOG.debug(f" - Adding stats from {fname}")
            
        with open(fname) as f:
            for line in f.readlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    LOG.warn(f" - {fname}: Ignoring line: {line}, did not contain key, value pair")
                else:
                    try:
                        subj_stats[parts[0]] = float(parts[1])
                    except ValueError:
                        if parts[1].strip() != "" and parts[0].strip().lower() != "subjid":
                           LOG.warn(f" - {fname}: Ignoring line: {line}, value was not numeric")

def add_csv_stats(subjid, subjdir, paths, subj_stats):
    for rel_path in paths:
        fglob = os.path.join(subjdir, rel_path)
        fnames = list(glob.glob(fglob))
        if not fnames:
            LOG.warn(f" - Failed to find any file matching {fglob}")
            return
        elif len(fnames) > 1:
            LOG.warn(f" - Multiple files matching {fglob} - will use first which is {fnames[0]}")
        fname = fnames[0]

        LOG.debug(f" - Adding CSV stats from {fname}")
        with open(fname) as f:
            lines = f.readlines()
            if len(lines) < 2:
                LOG.warn(f" - {fname}: Ignoring, does not contain keys, values on separate lines")
                continue
            elif len(lines) > 2:
                LOG.warn(f" - {fname}: contains more than 2 lines, ignoring extras")

            keys = [p.strip() for p in lines[0].split(",")]
            values = [p.strip() for p in lines[1].split(",")]
            if len(keys) != len(values):
                LOG.warn(f" - {fname}: Ignoring, keys and values have different lengths")
                continue
            
            for k, v in zip(keys, values):
                try:
                    subj_stats[k] = float(v)
                except ValueError:
                    if k.strip().lower() != "subjid":
                        LOG.warn(f" - {fname}: Ignoring key: {k}, value {v} was not numeric")
