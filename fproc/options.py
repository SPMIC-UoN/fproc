"""
FPROC: Base class for study specific pipeline options
"""
import argparse
import logging

from ._version import __version__

LOG = logging.getLogger(__name__)

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, name, version):
        argparse.ArgumentParser.__init__(self, f'{name} v{version} (using fproc v{__version__})', add_help=True)
        self.add_argument('--input', help='Path to input folder', required=True)
        self.add_argument('--output', help='Path to output folder', required=True)
        self.add_argument('--input-subfolder', help='Optional subfolder for pipeline input')
        self.add_argument('--output-subfolder', help='Optional subfolder for pipeline output')
        self.add_argument('--subjid', help='Subject ID')
        self.add_argument('--subjidx', type=int, help='Subject index, starting at zero. Subjects are identified by subdirectories of input')
        self.add_argument("--skip", help="Comma separated list of modules to skip", default="")
        self.add_argument('--overwrite', action="store_true", default=False, help='If specified, overwrite any existing output')
        self.add_argument('--debug', action="store_true", default=False, help='Enable debug output')

class StatsCombineArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="fproc_combine", add_help=True, **kwargs)
        self.add_argument("--input", help="Directory containing subject folders", required=True)
        self.add_argument("--output", help="Path to output CSV file", default="renal_preproc_stats.csv")
        self.add_argument("--path", nargs="+", default=[], help="Relative path to Key, Value output files for each subject (may be more than one)")
        self.add_argument("--path-csv", nargs="+", default=[], help="Relative path to CSV output files for each subject (may be more than one)")
        self.add_argument("--paths", help="Path to text file containing output file relative paths")
        self.add_argument("--paths-csv", help="Path to text file containing output file CSV output relative paths")
        self.add_argument("--subjids", help="Optional file containing subject IDs to process - if not specified will use all subdirs of input")
        self.add_argument("--skip-empty", action="store_true", default=False, help="If specified, columns will be ignored if there are no nonzero values for any subject")
        self.add_argument("--overwrite", help="Overwrite existing output", action="store_true", default=False)
        self.add_argument("--debug", help="Enable debugging output", action="store_true", default=False)

class FlattenArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        argparse.ArgumentParser.__init__(self, prog="fproc_flatten", add_help=True, **kwargs)
        self.add_argument("--input", help="Directory containing subject folders", required=True)
        self.add_argument("--output", help="Path to output folder", required=True)
        self.add_argument("--subjids", help="Optional file containing subject IDs to process - if not specified will use all subdirs of input")
        self.add_argument("--path", help="Relative path to preproc output", default=".")
        self.add_argument("--matcher", help="Optional matcher - only include PNG files containing this substring", default="")
        self.add_argument("--overwrite", help="Overwrite existing output", action="store_true", default=False)
        self.add_argument("--debug", help="Enable debugging output", action="store_true", default=False)
