import logging

import numpy as np
import scipy

from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, StatsModule

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class T1MulticonSeg(Module):
    def __init__(self):
        Module.__init__(self, "t1_seg")

    def process(self):
        t1map = self.inimg("t1multicon", "t1_multicon.nii.gz")
        LOG.info(f" - Segmenting multi-contrast T1 map from {t1map.fname} using model {self.pipeline.options.t1_model}")
        self.runcmd([
            'kidney_t1_seg',
            '--input', t1map.dirname,
            '--subjid', '',
            '--display-id', self.pipeline.options.subjid,
            '--t1', t1map.fname,
            '--model', self.pipeline.options.t1_model,
            '--multicon',
            '--noclean',
            '--output', self.outdir,
            '--outprefix', f'seg_kidney_multicon'],
            logfile=f'seg_kidney_multicon.log'
        )

MODULES = [
    T1MulticonSeg(),
]

class T1MulticonArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "t1_multicon", __version__)
        self.add_argument("--t1-model", help="Filename or URL for T1 segmentation model weights", default="")

class T1Multicon(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "t1_multicon", __version__, T1MulticonArgumentParser(), MODULES)

if __name__ == "__main__":
    T1Multicon().run()
