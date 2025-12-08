"""
FSORT: File pre-sorter for imaging processing pipelines
"""
import importlib
import os
import sys
import logging

from ._version import __version__
from .pipeline import Pipeline
from .options import ArgumentParser

LOG = logging.getLogger(__name__)

def _setup_logging(args):
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

def run(config_fname):
    sys.argv.append(f"--config={config_fname}")
    main()

def main():
    """
    FPROC command line entry point
    """
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    parser = ArgumentParser("fproc", __version__)
    options, _extras = parser.parse_known_args()
    _setup_logging(options)
    LOG.info(f"FPROC v{__version__}")

    try:
        LOG.info(f" - Loading configuration from {options.pipeline}")
        pipeline = importlib.import_module(options.pipeline)
        pipeline_config_fname = pipeline.__name__
    except ImportError:
        pipeline_config_fpath = os.path.abspath(os.path.normpath(options.pipeline))
        pipeline_config_dirname, pipeline_config_fname = os.path.split(pipeline_config_fpath)
        try:
            sys.path.append(pipeline_config_dirname)
            pipeline = importlib.import_module(pipeline_config_fname.replace(".py", ""))
        except ImportError:
            LOG.exception("Loading config")
            raise ValueError(f"Could not load configuration {options.pipeline} - must be a python module or file")
        finally:
            sys.path.remove(pipeline_config_dirname)

    if hasattr(pipeline, "add_options"):
        LOG.info(f" - Parsing pipeline-specific options")
        pipeline.add_options(parser)
        options = parser.parse_args()

    pipeline_name = getattr(pipeline, "NAME", pipeline_config_fname)
    pipeline_modules = getattr(pipeline, "MODULES", [])
    pipeline_version = getattr(pipeline, "__version__", "unknown")
    pipeline = Pipeline(pipeline_name, pipeline_version, options, pipeline_modules)
    pipeline.run()

if __name__ == "__main__":
    main()
