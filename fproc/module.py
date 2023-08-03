"""
FPROC: Base class for processing module
"""
import logging
import os

LOG = logging.getLogger(__name__)

class Module:
    """
    A processing module
    """
    
    def __init__(self, name):
        self.name = name
        self.pipeline = None

    def run(self, pipeline):
        self.pipeline = pipeline
        self.outdir = os.path.abspath(os.path.normpath(os.path.join(self.pipeline.options.output, self.name)))
        os.makedirs(self.outdir, exist_ok=True)
        self.process()

    def process(self):
        raise NotImplementedError()

    def indir(self, name):
        if self.pipeline is None:
            raise RuntimeError("No pipeline context")
        return os.path.abspath(os.path.normpath(os.path.join(self.pipeline.options.input, name)))
