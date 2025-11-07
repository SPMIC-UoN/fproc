"""
FPROC: Base class for imaging processing pipeline
"""

import datetime
import logging
import os
import sys

from ._version import __version__
from .module import ModuleError

LOG = logging.getLogger(__name__)


class Pipeline:

    def __init__(self, name, version, options, modules):
        self.name = name
        self.version = version
        try:
            # Compatibility while we migrate to built in main()
            self.options = options.parse_args()
            self._setup_logging()
        except:
            self.options = options
        self.modules = modules

    def run(self):
        try:
            # Identify the subject ID
            input_basename = os.path.basename(self.options.input)
            output_basename = os.path.basename(self.options.output)
            if self.options.subjid is not None and self.options.subjidx is not None:
                raise RuntimeError("Cannot specify subjid and subjidx at the same time")
            elif self.options.subjid is not None:
                subjdir_msg = f" - subjid is {self.options.subjid}"
            elif self.options.subjidx is not None:
                subjdirs = sorted(
                    [
                        d
                        for d in os.listdir(self.options.input)
                        if os.path.isdir(os.path.join(self.options.input, d))
                    ]
                )
                if self.options.subjidx < 0 or self.options.subjidx >= len(subjdirs):
                    raise RuntimeError(
                        f"Subject index out of range: {self.options.subjidx} - number of subjects found: {len(subjdirs)}"
                    )
                else:
                    self.options.subjid = subjdirs[self.options.subjidx]
                    subjdir_msg = f" - Processing subject index {self.options.subjidx}, subjid: {self.options.subjid}"
            else:
                self.options.subjid = output_basename
                subjdir_msg = (
                    f" - Setting subjid from output folder to {self.options.subjid}"
                )

            # Set up input/output directories
            if input_basename != self.options.subjid:
                self.options.input = os.path.join(
                    self.options.input, self.options.subjid
                )
            if output_basename != self.options.subjid:
                self.options.output = os.path.join(
                    self.options.output, self.options.subjid
                )
            if self.options.input_subfolder:
                self.options.input = os.path.join(
                    self.options.input, self.options.input_subfolder
                )
            if self.options.output_subfolder:
                self.options.output = os.path.join(
                    self.options.output, self.options.output_subfolder
                )

            if os.path.exists(self.options.output) and os.listdir(self.options.output):
                if self.options.overwrite:
                    outdir_msg = f" - Output directory {self.options.output} already exists - overwriting"
                else:
                    raise RuntimeError(
                        "Output directory {self.options.output} already exists - use --overwrite to replace"
                    )
            else:
                outdir_msg = f" - Output directory {self.options.output}"
                os.makedirs(self.options.output, exist_ok=True)
            self._start_logfile()

            banner = f"{self.name.upper()} v{self.version}"
            LOG.info("=" * len(banner))
            LOG.info(banner)
            LOG.info("=" * len(banner))
            timestamp = self.timestamp()
            LOG.info(f" - Start time {timestamp}")
            LOG.info(f" - Input directory {self.options.input}")
            LOG.info(outdir_msg)
            LOG.info(subjdir_msg)

            skip = [s.lower().strip() for s in self.options.skip.split(",")]
            skipdone = [s.lower().strip() for s in self.options.skipdone.split(",")]
            noskip = [s.lower().strip() for s in self.options.noskip.split(",")]
            autoskip = self.options.autoskip

            for module in self.modules:
                name = module.name.lower()
                last_done = module.timestamp(self.options.output)
                if name in noskip:
                    LOG.info(f"FORCING {module.name.upper()}")
                elif not last_done:
                    pass
                elif self.options.autoskip:
                    deps = [self._get_module(d) for d in module.deps]
                    if not deps:
                        LOG.info(f"SKIPPING {module.name.upper()} - ALREADY DONE")
                        continue
                    last_done_deps = [(d.name, d.timestamp(self.options.output)) for d in deps if d is not None]
                    out_of_date = [d for d in last_done_deps if d[1] is None or (last_done is not None and d[1] > last_done)]
                    if not out_of_date:
                        LOG.info(f"SKIPPING {module.name.upper()} - NOT OUT OF DATE")
                        continue
                    else:
                        LOG.info(f"OUT OF DATE {module.name.upper()}: {last_done} < {out_of_date}")
                elif name in skip:
                    LOG.info(f"SKIPPING {module.name.upper()}")
                    # Temporary to introduce the done.txt file retrospectively
                    module.outdir = os.path.abspath(
                        os.path.normpath(os.path.join(self.options.output, module.name))
                    )
                    self._write_done_file(module)
                    continue
                elif (
                    (name in skipdone or "*" in skipdone)
                    and last_done is not None
                ):
                    LOG.info(f"SKIPPING {module.name.upper()} - ALREADY DONE")
                    continue

                timestamp = self.timestamp()
                LOG.info(f"RUNNING {module.name.upper()} : last done {last_done} start time {timestamp}")
                try:
                    module.run(self)
                    timestamp = self._write_done_file(module)
                    LOG.info(f"DONE {module.name.upper()} : end time {timestamp}")
                except ModuleError as exc:
                    LOG.warn(f"FAILED {module.name.upper()}: {exc}")
                except:
                    LOG.exception(f"ERROR {module.name.upper()} - MODULE NOT COMPLETED")

            timestamp = self.timestamp()
            LOG.info(f"Finish time {timestamp}")
            LOG.info(f"{self.name.upper()} FINISHED")
        except:
            LOG.exception(f"{self.name.upper()} UNEXPECTED ERROR - STOPPING")
            sys.exit(1)

    def _write_done_file(self, module):
        timestamp = self.timestamp()
        done_file = os.path.join(module.outdir, "done.txt")
        with open(done_file, "w") as f:
            f.write(f"{module.name} completed at {timestamp}\n")
        LOG.info(f"Done file written: {done_file}")
        return timestamp

    def _get_module(self, name):
        for module in self.modules:
            if module.name.lower() == name.lower():
                return module
        return None

    def timestamp(self):
        return str(datetime.datetime.now())

    def _setup_logging(self, clear=True):
        if clear:
            logging.getLogger().handlers.clear()

        if self.options.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import warnings

            warnings.filterwarnings("ignore")

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)

    def _start_logfile(self):
        logfile = os.path.join(self.options.output, "logfile.txt")
        if os.path.exists(logfile):
            os.remove(logfile)
        handler = logging.FileHandler(logfile)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)
