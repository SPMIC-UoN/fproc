import glob
import logging
import os

from fproc.module import Module
from fproc.modules import segmentations
from fsort.image_file import ImageFile

LOG = logging.getLogger(__name__)

__version__ = "0.0.1"

NAME = "mongolia"

class T1Data(Module):
    def __init__(self, name="t1_data", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        srcdir = self.pipeline.options.src_t1_dir
        srcdir = os.path.join(srcdir, self.pipeline.options.subjid)
        LOG.info(f" - Looking for T1 data in {srcdir}")
        imgs = glob.glob(os.path.join(srcdir, "*.nii*"))
        for img in imgs:
            img = ImageFile(img, warn_json=False)
            img.save(self.outfile(img.fname))
            LOG.info(f" - Found T1 image {img.fname}")

MODULES = [
    T1Data(name="t1_data"),
    segmentations.TotalSeg(
        name="totalseg_pre",
        src_dir="t1_data",
        water_glob="*pret1vibe*",
        fat_glob=None,
        img_glob="*pret1vibe*",
    ),
    segmentations.TotalSeg(
        name="totalseg_post",
        src_dir="t1_data",
        water_glob="*postt1vibe*",
        fat_glob=None,
        img_glob="*postt1vibe*",
    ),
]

def add_options(parser):
    parser.add_argument("--src-t1-dir", help="Dir containing NIFTI T1 maps")
