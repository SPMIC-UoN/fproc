import logging

from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class T1(Module):
    def __init__(self):
        Module.__init__(self, "t1")

    def process(self):
        LOG.info(" - Processing T1 map")

MODULES = [
    T1(),
]

class RenalPreprocArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "renal_preproc", __version__)

class RenalPreproc(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "renal preproc", __version__, RenalPreprocArgumentParser(), MODULES)

if __name__ == "__main__":
    renal_preproc = RenalPreproc()
    renal_preproc.run()
