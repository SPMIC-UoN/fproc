"""
FPROC: Script to 'flatten' image output of preprocessing scripts as this can be a more convenient format for review
"""
import glob
import logging
import os
import shutil
import sys

from ._version import __version__
from .options import FlattenArgumentParser

LOG = logging.getLogger(__name__)

def main():
    arg_parser = FlattenArgumentParser()
    options = arg_parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    LOG.info(f"FPROC v{__version__} - OUTPUT COMBINE\n")
    LOG.info(f" - Input directory: {options.input}")
    LOG.info(f" - Output file: {options.output}")
    os.makedirs(options.output, exist_ok=True)

    if options.subjids:
        with open(options.subjids) as f:
            subjids = [l.strip() for l in f.readlines() if os.path.isdir(os.path.join(options.input, l.strip()))]
    else:
        subjids = sorted([d for d in os.listdir(options.input) if os.path.isdir(os.path.join(options.input, d))])

    for subjid in subjids:
        LOG.info(f" - Subject ID: {subjid}")
        subjdir = os.path.join(options.input, subjid)
        flatten_subject(subjid, subjdir, options.path, options.output, options.matcher)

def flatten_pngs(subjid, subjdir, outdir, matcher):
    for root, _dirs, files in os.walk(subjdir):
        for fname in files:
            if fname.endswith(".png") and (not matcher or matcher in fname):
                outname = os.path.join(outdir, f"{subjid}_{fname}")
                shutil.copyfile(os.path.join(root, fname), outname)

def flatten_subject(subjid, subjdir, path, outdir, matcher):
    if not os.path.isdir(subjdir):
        LOG.warn(f" - {subjdir} does not exist or is not a directory")
        return

    preproc_path = os.path.join(subjdir, path)
    if not os.path.exists(preproc_path):
        print(" - Preproc path not found - looking for multiple sessions")
        sessions = os.listdir(subjdir)
        for session in sessions:
            sessdir = os.path.join(subjdir, session)
            preproc_path = os.path.join(sessdir, path)
            if os.path.exists(preproc_path):
                print(f"   - Session {session}")
                flatten_pngs(f"{subjid}_{session}", sessdir, outdir, matcher)
    else:
        flatten_pngs(subjid, subjdir, outdir, matcher)
