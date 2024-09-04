import glob
import logging
import os

from fsort import ImageFile
from fproc.module import Module, StatsModule

import numpy as np

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class T2SimpleExp(Module):
    def __init__(self):
        Module.__init__(self, "t2_exp")

    def process(self):
        imgs = self.inimgs("t2", "t2_e*.nii.gz")
        if not imgs:
            self.no_data("No T2 mapping data found")
        elif len(imgs) not in (10, 11):
            self.bad_data(f"Incorrect number of T2 echos found: {len(imgs)}, expected 10 or 11")
        if len(imgs) == 11:
            LOG.warn("11 echos found - discarding last echo")
            imgs = imgs[:10]

        # Do this to make sure we get the echos in the correct order!
        imgs = [self.inimg("t2", f"t2_e_{echo}.nii.gz") for echo in range(1, 11)]

        # Import is expensive so delay until we need it
        from ukat.mapping.t2 import T2
        tes = np.array([i.echotime * 1000 for i in imgs]) # In milliseconds
        data = np.stack([i.data for i in imgs], axis=-1)
        first_echo = imgs[0]
        LOG.info(f" - Doing T2 mapping EXP fit on data shape {data.shape}, TEs: {tes}")
        mapper = T2(data, tes, first_echo.affine)
        LOG.info(f" - DONE T2 mapping EXP fit - saving")
        first_echo.save_derived(mapper.t2_map, self.outfile("t2_map.nii.gz"))
        first_echo.save_derived(mapper.m0_map, self.outfile("m0_map.nii.gz"))
        first_echo.save_derived(mapper.r2, self.outfile("r2_map.nii.gz"))
        LOG.info(f" - Saved data")

class T2Stim(Module):
    def __init__(self):
        Module.__init__(self, "t2_stim")

    def process(self):
        imgs = self.inimgs("t2", "t2_e*.nii.gz")
        if not imgs:
            self.no_data("No T2 mapping data found")
        elif len(imgs) not in (10, 11):
            self.bad_data(f"Incorrect number of T2 echos found: {len(imgs)}, expected 10 or 11")

        if len(imgs) == 11:
            LOG.warn("11 echos found - discarding last echo")
            imgs = imgs[:10]

        # Do this to make sure we get the echos in the correct order!
        imgs = [self.inimg("t2", f"t2_e_{echo}.nii.gz") for echo in range(1, 11)]

        # Import is expensive so delay until we need it
        from ukat.mapping.t2_stimfit import T2StimFit, StimFitModel
        first_echo = imgs[0]
        model = StimFitModel(ukrin_vendor=first_echo.vendor.lower())
        #tes = np.array([i.echotime for i in imgs])
        data = np.stack([i.data for i in imgs], axis=-1)
        LOG.info(f" - Doing T2 mapping STIM fit on data shape {data.shape}, vendor: {first_echo.vendor.lower()}")
        mapper = T2StimFit(data, first_echo.affine, model)
        LOG.info(f" - DONE T2 mapping STIM fit - saving")
        first_echo.save_derived(mapper.t2_map, self.outfile("t2_map.nii.gz"))
        first_echo.save_derived(mapper.m0_map, self.outfile("m0_map.nii.gz"))
        first_echo.save_derived(mapper.r2_map, self.outfile("r2_map.nii.gz"))
        first_echo.save_derived(mapper.b1_map, self.outfile("b1_map.nii.gz"))
        LOG.info(f" - Saved data")


class T1Segs(Module):
    def __init__(self):
        Module.__init__(self, "t1_segs")

    def process(self):
        seg_origs = self.inimgs("../t1_clean_out", "*.nii.gz")
        if not seg_origs:
            LOG.info(" - No automatic kidney segmentations found")

        for part in ("cortex", "medulla"):
            use_orig = True
            if not self.pipeline.options.kidney_masks:
                LOG.info(f" - No kidney masks dir specified - will use originals for {part}")
            else:
                globexpr = os.path.join(
                    self.pipeline.options.kidney_masks,
                    self.pipeline.options.subjid,
                    f"*{part}*.nii*"
                )
                masks = list(glob.glob(globexpr))
                if not masks:
                    LOG.info(f" - No {part} mask for {self.pipeline.options.subjid} in {globexpr} - will use originals")
                elif len(masks) != 2:
                    LOG.warn(f" - {len(masks)} {part} masks found for {self.pipeline.options.subjid}: {masks} - expected 2 (left/right) - ignoring")
                else:
                    left = [f for f in masks if "left" in f.lower()]
                    right = [f for f in masks if "right" in f.lower()]
                    if not left or not right:
                        LOG.warn(f" Did not find left+right {part} masks in: {masks} - ignoring")
                    else:
                        LOG.info(f" - Replacing original {part} masks with new ones: {left}, {right}")
                        use_orig = False

            if use_orig:
                left = [i for i in seg_origs if part in i.fname.lower() and "_l_" in i.fname.lower()]
                right = [i for i in seg_origs if part in i.fname.lower() and "_r_" in i.fname.lower()]
                if not left or not right:
                    LOG.warn(f"No original masks found for {part} - have no kidney segmentation")
                    continue
                elif len(left) > 1 or len(right) > 1:
                    LOG.warn(f"Multiple original masks found for {part} - will use first")
                left = left[0]
                right = right[0]
                LOG.info(f" - Using original masks: {left.fname}, {right.fname}")
            else:
                left = ImageFile(left[0], warn_json=False)
                right = ImageFile(right[0], warn_json=False)
                LOG.info(f" - Using manual masks: {left.fname}, {right.fname}")

            left.save(self.outfile(f"kidney_{part}_l.nii.gz"))
            right.save(self.outfile(f"kidney_{part}_r.nii.gz"))
            t1_maps = self.inimgs("molli_kidney", "t1_map.nii.gz")
            if t1_maps:
                self.lightbox(t1_maps[0], left, name=f"kidney_{part}_l_lightbox", tight=True)
                self.lightbox(t1_maps[0], right, name=f"kidney_{part}_r_lightbox", tight=True)

class Stats(StatsModule):
    def __init__(self):
        StatsModule.__init__(
            self, name="stats", 
            segs={
                "cortex_r" : {
                    "dir" : "t1_segs",
                    "glob" : "*cortex_r*.nii.gz"
                },
                "cortex_l" : {
                    "dir" : "t1_segs",
                    "glob" : "*cortex_l*.nii.gz"
                },
                # "cortex" : {
                #     "dir" : "t1_segs",
                #     "glob" : "*cortex_orig*.nii.gz"
                # },
                "medulla_r" : {
                    "dir" : "t1_segs",
                    "glob" : "*medulla_r*.nii.gz"
                },
                "medulla_l" : {
                    "dir" : "t1_segs",
                    "glob" : "*medulla_l*.nii.gz"
                },
                # "medulla" : {
                #     "dir" : "t1_segs",
                #     "glob" : "*medulla_orig*.nii.gz"
                # },
                "kidney_l" : {
                    "dir" : "t1_segs",
                    "glob" : "*all_l_orig*.nii.gz"
                },
                "kidney_r" : {
                    "dir" : "t1_segs",
                    "glob" : "*all_r_orig*.nii.gz"
                },
            },
            params={
                "t2_exp" : {
                    "dir" : "t2_exp",
                    "glob" : "t2_map.nii.gz",
                    "segs" : ["cortex_r", "cortex_l", "medulla_r", "medulla_l"],
                },
                "t2_stim" : {
                    "dir" : "t2_stim",
                    "glob" : "t2_map.nii.gz",
                    "segs" : ["cortex_r", "cortex_l", "medulla_r", "medulla_l"],
                },
                "b1_stim" : {
                    "dir" : "t2_stim",
                    "glob" : "b1_map.nii.gz",
                }
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
            seg_volumes=False,
        )

NAME="tk21_t2"

MODULES = [
    T2SimpleExp(),
    T2Stim(),
    T1Segs(),
    Stats(),
]
