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
        self.add_argument('--overwrite', action="store_true", default=False, help='If specified, overwrite any existing output')
        self.add_argument('--debug', action="store_true", default=False, help='Enable debug output')
