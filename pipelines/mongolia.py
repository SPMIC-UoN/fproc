import glob
import logging
import os

from fproc.module import Module
from fproc.modules import segmentations
from fsort.image_file import ImageFile

LOG = logging.getLogger(__name__)

__version__ = "0.0.1"

NAME = "mongolia"
STUDYDIR = "/gpfs01/spmstore/project/RenalMRI/mongolia"
OUTNAME = NAME

BATCH = {
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
        "totalseg-img",
    ],
    "time": "24:00:00",
    "mem_per_cpu": "32g",
    "subjects": 230,
    "input_subfolder": "",
    "output_subfolder": "",
    "extra_fproc_args": [
        "--src-t1-dir=/spmstore/project/RenalMRI/Mongolia_Onom/t1vibe_pre_and_post",
    ],
}

COHORTS = [
    ("full cohort", "", None),
]

OUTFILES = {
    "totalseg_pre": [
        "fproc/totalseg_pre/volumes.csv",
    ],
    "totalseg_post": [
        "fproc/totalseg_post/volumes.csv",
    ],
}

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
