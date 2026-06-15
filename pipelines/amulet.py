# AMULET pipeline

import logging
import os

import numpy as np

from fsort.image_file import ImageFile
from fproc.module import Module
from fproc.modules import (
    maps,
    segmentations,
    statistics,
    seg_postprocess,
    regrid,
    align,
    misc,
)

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME = "amulet"


class T1wFatWater(Module):
    """
    Create fat and water images from t1w dual echo
    """

    def __init__(self, **kwargs):
        super().__init__(name="t1w_fat_water", **kwargs)

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
                water_calc, self.outfile("t1w_water_calc.nii.gz")
            )
            fat_calc = (inphase.data - opphase.data) / 2
            fat_calc = inphase.save_derived(
                fat_calc, self.outfile("t1w_fat_calc.nii.gz")
            )

        if water_in and fat_in:
            LOG.info(" - Using scanner fat/water outputs as default")
            water_in.save(self.outfile("t1w_water.nii.gz"))
            fat_in.save(self.outfile("t1w_fat.nii.gz"))
        elif inphase and opphase:
            LOG.info(" - Using calculated fat/water outputs as default")
            water_calc.save(self.outfile("t1w_water.nii.gz"))
            fat_calc.save(self.outfile("t1w_fat.nii.gz"))
        else:
            self.no_data(
                "No input for fat/water calculation; no scanner fat/water either"
            )


MODULES = [
    misc.ScanDates(
        "scan_dates",
        input={
            "../fsort/t1w_in": "*.nii.gz",
            "../fsort/t2w_fs": "*.nii.gz",
        },
    ),
    T1wFatWater(),
    maps.FatFractionDixon(
        dixon_dir="../fproc/t1w_fat_water",
        ff_name="fat_fraction",
        ff_calc_name="fat_fraction_calc",
        fat_name="t1w_fat",
        water_name="t1w_water",
    ),
    segmentations.TotalSeg(
        name="totalseg_t1w",
        src_dir="../fproc/t1w_fat_water",
        water_glob="t1w_water.nii.gz",
        fat_glob="t1w_fat.nii.gz",
    ),
    segmentations.TotalSeg(
        name="totalseg_t2w",
        src_dir="../fsort/t2w_fs",
        water_glob="t2w_fs.nii.gz",
        fat_glob=None,
    ),
    segmentations.TotalSeg(
        name="totalseg_dwi",
        src_dir="../fsort/dwi_adc",
        water_glob="dwi_adc.nii.gz",
        fat_glob=None,
    ),
]
