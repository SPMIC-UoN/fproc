import logging
import os
import glob

from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module

from fsort.image_file import ImageFile

import numpy as np

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class DiffusionOverlays(Module):
    def __init__(self):
        Module.__init__(self, "diffusion_overlays")

    def process(self):
        diff_img = self.inimg("diffusion", "diffusion_firstvol.nii.gz")
        LOG.info(f" - Found diffusion first volume: {diff_img.fname}")
        preproc_output = os.path.join(self.pipeline.options.preproc_output, self.pipeline.options.subjid)
        LOG.info(f" - Using preprocessing output from: {preproc_output}")
        cortex_masks = list(glob.glob(os.path.join(preproc_output, "t1_out", "seg_kidney_*_cortex_t1.nii.gz")))
        medulla_masks = list(glob.glob(os.path.join(preproc_output, "t1_out", "seg_kidney_*_medulla_t1.nii.gz")))
        if not cortex_masks:
            self.no_data("No cortex masks found")
        if not medulla_masks:
            self.no_data("No medulla masks found")
        cortex_mask_res = np.zeros(diff_img.shape3d, dtype=int)
        medulla_mask_res = np.zeros(diff_img.shape3d, dtype=int)
        for mask in cortex_masks:
            mask = ImageFile(mask)
            mask = self.resample(mask, diff_img, allow_rotated=True, is_roi=True)
            cortex_mask_res += mask.get_fdata().astype(int)
        for mask in medulla_masks:
            mask = ImageFile(mask)
            mask = self.resample(mask, diff_img, allow_rotated=True, is_roi=True)
            medulla_mask_res += mask.get_fdata().astype(int)
        self.lightbox(diff_img.data, cortex_mask_res, "diff_firstvol_cortex")
        self.lightbox(diff_img.data, medulla_mask_res, "diff_firstvol_medulla")

MODULES = [
    DiffusionOverlays(),
]

class EmpaDiffusionArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "empa_diffusion", __version__)
        self.add_argument("--preproc-output", required=True)
        
class EmpaDiffusion(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "empa_diffusion", __version__, EmpaDiffusionArgumentParser(), MODULES)

if __name__ == "__main__":
    EmpaDiffusion().run()
