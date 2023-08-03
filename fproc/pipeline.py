"""
FPROC: Base class for imaging processing pipeline
"""
import datetime
import logging
import os
import sys

from ._version import __version__

LOG = logging.getLogger(__name__)

class Pipeline:

    def __init__(self, name, version, option_parser, modules):
        self.name = name
        self.version = version
        self.options = option_parser.parse_args()
        self.modules = modules
        self._setup_logging()

    def run(self):
        try:
            banner = f"{self.name.upper()} v{self.version} (using fproc v{__version__})"
            LOG.info(banner)
            LOG.info("=" * len(banner))
            timestamp = self.timestamp()
            LOG.info(f" - Start time {timestamp}")
            LOG.info(f" - Input directory {self.options.input}")
            if os.path.exists(self.options.output):
                if self.options.overwrite:
                    LOG.warn(f" - Output directory {self.options.output} already exists - overwriting")
                else:
                    raise RuntimeError(f"Output directory {self.options.output} already exists - use --overwrite to replace")
            else:
                LOG.info(f" - Output directory {self.options.output}")
                os.makedirs(self.options.output)

            for module in self.modules:
                timestamp = self.timestamp()
                LOG.info(f"Running module {module.name} - start time {timestamp}")
                module.run(self)
                timestamp = self.timestamp()
                LOG.info(f"DONE running module {module.name} - end time {timestamp}")

            timestamp = self.timestamp()
            LOG.info(f"Finish time {timestamp}")
            LOG.info(f"DONE {self.name.upper()}")
        except:
            LOG.exception("Unexpected error")
            sys.exit(1)

    def timestamp(self):
        return str(datetime.datetime.now())

    def _setup_logging(self):
        if self.options.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
