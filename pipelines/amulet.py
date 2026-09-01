# AMULET pipeline

import logging
import os

import numpy as np

from fproc.module import Module
from fproc.modules import (
    maps,
    segmentations,
    statistics,
    misc,
    qa,
    magorino,
)

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

# Configuration
NAME = "amulet"
STUDYDIR = os.path.join("/gpfs01/spmstore/project/LiverMRI", NAME)
OUTNAME = NAME

COHORTS = [
    ("full cohort", "", None),
]

OUTFILES = {
    "liver": [
        "fproc/stats/stats.csv",
        "fproc/isnr/isnr.csv",
    ],
    "liver_shape": [
        "fproc/t1w_core_liver_shape/radiomics.csv",
        "fproc/t1w_fs_core_liver_shape/radiomics.csv",
        "fproc/t2w_core_liver_shape/radiomics.csv",
        "fproc/t2w_fs_core_liver_shape/radiomics.csv",
        "fproc/dwi_core_liver_shape/radiomics.csv"
    ],
    "liver_radiomics": [
        "fproc/t1w_core_radiomics/radiomics.csv",
        "fproc/t1w_fs_core_radiomics/radiomics.csv",
        "fproc/t2w_core_radiomics/radiomics.csv",
        "fproc/t2w_fs_core_radiomics/radiomics.csv",
        "fproc/dwi_core_radiomics/radiomics.csv",
    ],
    "qa": [
        "fproc/qa/qa.csv",
    ],
}

ALLOW_TEXT = ["qa"]

class T1wFatWater(Module):
    """
    Create fat and water images from t1w dual echo
    """

    def __init__(self, **kwargs):
        super().__init__(name="t1w_core_fat_water", **kwargs)

    def process(self):
        water_in = self.single_inimg(
            "../fsort/t1w_water", "t1w_water.nii.gz", warn=False
        )
        fat_in = self.single_inimg("../fsort/t1w_fat", "t1w_fat.nii.gz", warn=False)
        inphase = self.single_inimg("../fsort/t1w_in", "t1w_in.nii.gz", warn=False)
        opphase = self.single_inimg("../fsort/t1w_out", "t1w_out.nii.gz", warn=False)
        if water_in and fat_in:
            LOG.info(" - Found scanner generated fat/water outputs")
            water_in.save(self.outfile("t1w_water_scanner.nii.gz"))
            fat_in.save(self.outfile("t1w_fat_scanner.nii.gz"))

        if inphase and opphase:
            LOG.info(" - Found dual echo in/opp: calculating fat/water maps")
            water_calc = (inphase.data + opphase.data) / 2
            water_calc = inphase.save_derived(
                water_calc, self.outfile("t1w_core_water_calc.nii.gz")
            )
            fat_calc = (inphase.data - opphase.data) / 2
            fat_calc = inphase.save_derived(
                fat_calc, self.outfile("t1w_core_fat_calc.nii.gz")
            )

        if water_in and fat_in:
            LOG.info(" - Using scanner fat/water outputs as default")
            water_in.save(self.outfile("t1w_core_water.nii.gz"))
            fat_in.save(self.outfile("t1w_core_fat.nii.gz"))
        elif inphase and opphase:
            LOG.info(" - Using calculated fat/water outputs as default")
            water_calc.save(self.outfile("t1w_core_water.nii.gz"))
            fat_calc.save(self.outfile("t1w_core_fat.nii.gz"))
        else:
            self.no_data(
                "No input for fat/water calculation; no scanner fat/water either"
            )

class DwiB0(Module):
    """
    Create b0 image from dwi
    """

    def __init__(self, **kwargs):
        super().__init__(name="dwi_core_minbval", **kwargs)

    def process(self):
        dwi_in = self.single_inimg("../fsort/dwi", "dwi.nii.gz", warn=False)
        if not dwi_in:
            self.bad_data("No DWI input found")
        elif dwi_in.bval is None or len(dwi_in.bval) == 0:
            self.bad_data("No bval file found for DWI input")
        b0_vols = [idx for idx in range(dwi_in.data.shape[-1]) if np.isclose(dwi_in.bval[idx], 0)]
        if not b0_vols:
            min_bval = np.min(dwi_in.bval)
            LOG.warning(f"No b0 volumes found in dwi input; using min bval {min_bval} instead")
            b0_vols = [idx for idx in range(dwi_in.data.shape[-1]) if np.isclose(dwi_in.bval[idx], min_bval)]

        if len(b0_vols) > 1:
            LOG.warning(f"Found {len(b0_vols)} min-bval volumes in dwi input; using first")

        b0_data = dwi_in.data[..., b0_vols[0]]
        LOG.info(" - Using volume {} for b0 calculation".format(b0_vols[0]))
        dwi_in.save_derived(b0_data, self.outfile("dwi_core_minbval.nii.gz"))


MODULES = [
    qa.QA(config="/spmstore/project/LiverMRI/amulet/qa_config.xlsx"),
    misc.ScanDates(
        "scan_dates",
        input={
            "../fsort/t1w_in": "*.nii.gz",
            "../fsort/t2w_fs": "*.nii.gz",
        },
    ),
    T1wFatWater(),
    DwiB0(),
    magorino.Magorino(
        t2star_dir="../fsort/t2star",
        t2star_glob="t2star_e_*.nii.gz",
    ),
    maps.FatFractionDixon(
        dixon_dir="t1w_core_fat_water",
        dixon_src="OUTPUT",
        ff_name="fat_fraction",
        ff_calc_name="fat_fraction_calc",
        fat_name="t1w_core_fat",
        water_name="t1w_core_water",
    ),
    segmentations.TotalSeg(
        name="t1w_core_totalseg",
        src_dir="../fproc/t1w_core_fat_water",
        water_glob="t1w_core_water.nii.gz",
        fat_glob=None,
    ),
    segmentations.TotalSeg(
        name="t1w_fs_core_totalseg",
        src_dir="../fsort/t1w_fs",
        water_glob="t1w_fs.nii.gz",
        fat_glob=None,
    ),
    segmentations.TotalSeg(
        name="t2w_core_totalseg",
        src_dir="../fsort/t2w_core",
        water_glob="t2w_core.nii.gz",
        fat_glob=None,
    ),
    segmentations.TotalSeg(
        name="t2w_fs_core_totalseg",
        src_dir="../fsort/t2w_fs",
        water_glob="t2w_fs.nii.gz",
        fat_glob=None,
    ),
    segmentations.TotalSeg(
        name="dwi_core_totalseg",
        src_dir="../fproc/dwi_core_minbval",
        water_glob="dwi_core_minbval.nii.gz",
        fat_glob=None,
    ),
    statistics.Radiomics(
        name="t1w_core_liver_shape",
        deps=["t1w_core_totalseg"],
        params={
            "water": {"dir": "t1w_core_fat_water", "fname": "t1w_core_water.nii.gz"},
            "fat": {"dir": "t1w_core_fat_water", "fname": "t1w_core_fat.nii.gz"},
        },
        segs={
            "liver": {"dir": "t1w_core_totalseg", "fname": "liver.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="t1w_fs_core_liver_shape",
        deps=["t1w_fs_core_totalseg"],
        params={
            "t1w": {"dir": "../fsort/t1w_fs", "fname": "t1w_fs.nii.gz"},
        },
        segs={
            "liver": {"dir": "t1w_fs_core_totalseg", "fname": "liver.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="t2w_core_liver_shape",
        deps=["t2w_core_totalseg"],
        params={
            "t2w_core": {"dir": "../fsort/t2w_core", "fname": "t2w_core.nii.gz"},
        },
        segs={
            "liver": {"dir": "t2w_core_totalseg", "fname": "liver.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="t2w_fs_core_liver_shape",
        deps=["t2w_fs_core_totalseg"],
        params={
            "t2w_fs": {"dir": "../fsort/t2w_fs", "fname": "t2w_fs.nii.gz"},
        },
        segs={
            "liver": {"dir": "t2w_fs_core_totalseg", "fname": "liver.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="dwi_core_liver_shape",
        deps=["dwi_core_totalseg"],
        params={
            "dwi_core_minbval": {"dir": "dwi_core_minbval", "fname": "dwi_core_minbval.nii.gz"},
        },
        segs={
            "liver": {"dir": "dwi_core_totalseg", "fname": "liver.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="t1w_core_radiomics",
        deps=["t1w_core_fat_water", "t1w_core_totalseg"],
        params={
            "t1w_water": {"dir": "t1w_core_fat_water", "fname": "t1w_core_water.nii.gz"},
            "t1w_fat": {"dir": "t1w_core_fat_water", "fname": "t1w_core_fat.nii.gz"},
            "t1w_ip": {"dir": "../fsort/t1w_in", "fname": "t1w_in.nii.gz"},
            "t1w_op": {"dir": "../fsort/t1w_out", "fname": "t1w_out.nii.gz"},
            "t1w_ff": {"dir": "fat_fraction", "fname": "fat_fraction_calc.nii.gz"},
        },
        segs={
            "liver": {"dir": "t1w_core_totalseg", "fname": "liver.nii.gz"},
        },
        image_types=["Original"],
    ),
    statistics.Radiomics(
        name="t1w_fs_core_radiomics",
        deps=["t1w_fs_core_totalseg"],
        params={
            "t1w_fs": {"dir": "../fsort/t1w_fs", "fname": "t1w_fs.nii.gz"},
        },
        segs={
            "liver": {"dir": "t1w_fs_core_totalseg", "fname": "liver.nii.gz"},
        },
        image_types=["Original"],
    ),
    statistics.Radiomics(
        name="t2w_core_radiomics",
        deps=["t2w_core_totalseg"],
        params={
            "t2w_core": {"dir": "../fsort/t2w_core", "fname": "t2w_core.nii.gz"},
        },
        segs={
            "liver": {"dir": "t2w_core_totalseg", "fname": "liver.nii.gz"},
        },
        image_types=["Original"],
    ),
    statistics.Radiomics(
        name="t2w_fs_core_radiomics",
        deps=["t2w_fs_core_totalseg"],
        params={
            "t2w_fs": {"dir": "../fsort/t2w_fs", "fname": "t2w_fs.nii.gz"},
        },
        segs={
            "liver": {"dir": "t2w_fs_core_totalseg", "fname": "liver.nii.gz"},
        },
        image_types=["Original"],
    ),
    statistics.Radiomics(
        name="dwi_core_radiomics",
        deps=["dwi_core_totalseg"],
        params={
            "dwi_adc": {"dir": "../fsort/dwi_adc", "fname": "dwi_adc.nii.gz"},
        },
        segs={
            "liver": {"dir": "dwi_core_totalseg", "fname": "liver.nii.gz"},
        },
        image_types=["Original"],
    ),
    statistics.SegStats(
        name="stats",
        default_limits="3t",
        segs={
            "liver_t1w_core": {
                "dir": "t1w_core_totalseg",
                "glob": "liver.nii.gz",
                "params" : ["ff_core"],
            },
            "liver_t1w_fs_core": {
                "dir": "t1w_fs_core_totalseg",
                "glob": "liver.nii.gz",
                "params" : [],
            },
            "liver_t2w_core": {
                "dir": "t2w_core_totalseg",
                "glob": "liver.nii.gz",
                "params" : [],
            },
            "liver_t2w_fs_core": {
                "dir": "t2w_fs_core_totalseg",
                "glob": "liver.nii.gz",
                "params" : [],
            },
            "liver_dwi_core": {
                "dir": "dwi_core_totalseg",
                "glob": "liver.nii.gz",
                "params" : ["adc_core"],
            },
        },
        params={
            "ff_core": {
                "dir": "fat_fraction",
                "glob": "fat_fraction_calc.nii.gz",
            },
            "adc_core": {
                "dir": "../fsort/dwi_adc",
                "glob": "dwi_adc.nii.gz",
            },
        },
        stats=[
            "n",
            "vol",
            "iqn",
            "iqvol",
            "iqmean",
            "median",
            "iqstd",
            "perc90",
            "mode",
            "fwhm",
        ],
        seg_volumes=True,
    ),
    statistics.ISNR(
        src=Module.INPUT,
        imgs={
            "t1w_in": "t1w_in.nii.gz",
            "t1w_out": "t1w_out.nii.gz",
            "t1w_fs": "t1w_fs.nii.gz",
            "t2w_core": "t2w_core.nii.gz",
            "t2w_fs": "t2w_fs.nii.gz",
            "dwi_adc": "dwi_adc.nii.gz",
            "../fproc/dwi_core_minbval" : "dwi_core_minbval.nii.gz",
        },
    ),
]
