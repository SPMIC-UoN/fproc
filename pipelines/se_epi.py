import glob
import logging
import os

import numpy as np
import nibabel as nib

from fsort import ImageFile
from fproc.module import Module
from fproc.modules import statistics

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class Nifti(Module):
    def __init__(self):
        Module.__init__(self, "nifti")

    def process(self):
        datadir = self.pipeline.options.datadir
        subjid = self.pipeline.options.subjid

        overlay_masks = {}
        for mask in ("cortex", "medulla"):
            masks = list(glob.glob(os.path.join(datadir, "masks", f"{subjid}_*{mask}*.nii*")))
            if not masks:
                LOG.warn(f"No {mask} masks found for {subjid}")
            elif len(masks) == 2:
                LOG.info(f" - L/R {mask} mask found for {subjid}")
                mask_l = [f for f in masks if "L" in f]
                mask_r = [f for f in masks if "R" in f]
                if not mask_l or not mask_r:
                    LOG.warn(f"Could not identify L/R {mask} masks for {subjid}: {masks} - ignoring")
                else:
                    left = ImageFile(mask_l[0], warn_json=False)
                    left.save(self.outfile(f"{mask}_l.nii.gz"))
                    right = ImageFile(mask_r[0], warn_json=False)
                    right.save(self.outfile(f"{mask}_r.nii.gz"))
                    combined_data = np.logical_or(left.data, right.data).astype(np.int32)
                    right.save_derived(combined_data, self.outfile(f"{mask}.nii.gz"))
                    overlay_masks[mask] = ImageFile(self.outfile(f"{mask}.nii.gz"), warn_json=False)
            elif len(masks) == 1:
                LOG.info(f" - Single {mask} mask found for {subjid}: {masks[0]} - splitting L/R")
                combined = ImageFile(masks[0])
                combined.save(self.outfile(f"{mask}.nii.gz"))
                overlay_masks[mask] = combined
                ax_labels = nib.orientations.aff2axcodes(combined.affine)
                slices_l, slices_r = [None] * 3, [None] * 3
                if "R" in ax_labels:
                    lr_axis = ax_labels.index("R")
                    lr_width = combined.shape[lr_axis]
                    lr_centre = int(lr_width/2)
                    LOG.info(f" - LR axis is {lr_axis} length {lr_width}")
                    slices_l[lr_axis] = slice(lr_centre, lr_width-1)
                    slices_r[lr_axis] = slice(0, lr_centre)
                elif "L" in ax_labels:
                    rl_axis = ax_labels.index("L")
                    rl_width = combined.shape[rl_axis]
                    rl_centre = int(rl_width/2)
                    LOG.info(f" - RL axis is {rl_axis} length {rl_width}")
                    slices_r[rl_axis] = slice(rl_centre, rl_width-1)
                    slices_l[rl_axis] = slice(0, rl_centre)
                else:
                    LOG.warn("Could not identify LR axis - ignoring")
                    return

                mask_l = np.copy(combined.data > 0).astype(np.int32)
                mask_r = np.copy(combined.data > 0).astype(np.int32)
                mask_l[tuple(slices_l)] = 0
                mask_r[tuple(slices_r)] = 0
                combined.save_derived(mask_l, self.outfile(f"{mask}_l.nii.gz"))
                combined.save_derived(mask_r, self.outfile(f"{mask}_r.nii.gz"))
                left = ImageFile(self.outfile(f"{mask}_l.nii.gz"), warn_json=False)
                right = ImageFile(self.outfile(f"{mask}_r.nii.gz"), warn_json=False)
            else:
                LOG.warn(f"Wrong number of {mask} masks for {subjid}: {masks} - ignoring")        

        for name, fname, subdir, matcher_glob in [
            ("T1", "t1", "T1_maps", f"{subjid}_t1*.nii*"),
            ("T1 3-param", "t1_3p", "T1_maps_3param", f"{subjid}_3p_t1*.nii*"),
            ("pCASL", "asl_pcasl", "ASL", f"{subjid}_*pCASL*perc_change*.nii*"),
            ("FAIR", "asl_fair", "ASL", f"{subjid}_*FAIR*perc_change*.nii*"),
            ("ADC", "adc", "DWI", f"{subjid}_ADCmap.nii*"),
        ]:
            imgs = list(glob.glob(os.path.join(datadir, subdir, matcher_glob)))
            if not imgs:
                LOG.warn(f"No {name} found for {subjid}")
            else:
                if len(imgs) > 1:
                    LOG.warn(f"Multiple {name}s found for {subjid}: {imgs} - choosing first")
                img = imgs[0]
                LOG.info(f" - Using {name}: {img}")
                map = ImageFile(img, warn_json=False)
                map.save(self.outfile(f"{fname}.nii.gz"))
                for mask in ("cortex", "medulla"):
                    if mask in overlay_masks:
                        self.lightbox(map, overlay_masks[mask], self.outfile(f"{fname}_{mask}_lightbox"))

class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats", 
            segs={
                "kidney_cortex_l" : {
                    "dir" : "nifti",
                    "glob" : "cortex_l.nii.gz"
                },
                "kidney_cortex_r" : {
                    "dir" : "nifti",
                    "glob" : "cortex_r.nii.gz"
                },
                "kidney_medulla_l" : {
                    "dir" : "nifti",
                    "glob" : "medulla_l.nii.gz"
                },
                "kidney_medulla_r" : {
                    "dir" : "nifti",
                    "glob" : "medulla_r.nii.gz"
                },
            },
            params={
                "t1" : {
                    "dir" : "nifti",
                    "glob" : "t1.nii.gz",
                },
                "t1_3p" : {
                    "dir" : "nifti",
                    "glob" : "t1_3p.nii.gz",
                },
                "asl_pcasl" : {
                    "dir" : "nifti",
                    "glob" : "asl_pcasl.nii.gz",
                },
                "asl_fair" : {
                    "dir" : "nifti",
                    "glob" : "asl_fair.nii.gz",
                },
                "adc" : {
                    "dir" : "nifti",
                    "glob" : "adc.nii.gz",
                },
            },
            stats=["iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

NAME = "se_epi"

__version__ = "0.0.1"

MODULES = [
    Nifti(),
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--datadir", help="Input data dir")
