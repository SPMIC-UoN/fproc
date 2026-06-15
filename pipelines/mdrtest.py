import logging
import os

from fsort import ImageFile
from fproc.module import Module
from fproc.modules import statistics, maps

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)


class Nifti(Module):
    def __init__(self):
        Module.__init__(self, "nifti")

    def process(self):
        datadir = self.pipeline.options.datadir
        subjid = self.pipeline.options.subjid

        t1 = ImageFile(os.path.join(datadir, f"{subjid}.nii.gz"))
        t1.save(self.outfile("t1.nii.gz"))


class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats",
            segs={
                "kidney_cortex_l": {"dir": "nifti", "glob": "cortex_l.nii.gz"},
                "kidney_cortex_r": {"dir": "nifti", "glob": "cortex_r.nii.gz"},
                "kidney_medulla_l": {"dir": "nifti", "glob": "medulla_l.nii.gz"},
                "kidney_medulla_r": {"dir": "nifti", "glob": "medulla_r.nii.gz"},
            },
            params={
                "t1": {
                    "dir": "nifti",
                    "glob": "t1.nii.gz",
                },
                "t1_3p": {
                    "dir": "nifti",
                    "glob": "t1_3p.nii.gz",
                },
                "asl_pcasl": {
                    "dir": "nifti",
                    "glob": "asl_pcasl.nii.gz",
                },
                "asl_fair": {
                    "dir": "nifti",
                    "glob": "asl_fair.nii.gz",
                },
                "adc": {
                    "dir": "nifti",
                    "glob": "adc.nii.gz",
                },
            },
            stats=["iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )


NAME = "mdrtest"

__version__ = "0.0.1"

MODULES = [
    Nifti(),
    maps.T1Molli(
        name="t1_nomdr",
        molli_dir="nifti",
        molli_glob="t1.nii.gz",
        mdr=False,
        use_scanner_maps=False,
        tis=[
            110,
            210,
            310,
            410,
            510,
            610,
            710,
            810,
            910,
            1010,
            1110,
            1210,
            1310,
            1410,
            1510,
        ],
        tis_use_md=False,
    ),
    maps.T1Molli(
        name="t1_mdr",
        molli_dir="nifti",
        molli_glob="t1.nii.gz",
        mdr=True,
        use_scanner_maps=False,
        tis=[
            110,
            210,
            310,
            410,
            510,
            610,
            710,
            810,
            910,
            1010,
            1110,
            1210,
            1310,
            1410,
            1510,
        ],
        tis_use_md=False,
    ),
    # SegStats(),
]


def add_options(parser):
    parser.add_argument("--datadir", help="Input data dir")
