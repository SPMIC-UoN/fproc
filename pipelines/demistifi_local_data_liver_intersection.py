import logging
import os

import numpy as np

from fproc.module import Module
from fproc.modules import segmentations, seg_postprocess, statistics

LOG = logging.getLogger(__name__)

class T2Star(Module):
    def __init__(self):
        Module.__init__(self, "t2star")

    def process(self):
        imgs = self.copyinput("dixon", "t2star.nii.gz")
        if imgs:
            # 100 is a fill value - replace with something easier to exclude in stats
            LOG.info(" - Saving T2* map with excluded fill value")
            exclude_fill = np.copy(imgs[0].data)
            exclude_fill[np.isclose(exclude_fill, 100)] = -9999
            imgs[0].save_derived(exclude_fill, self.outfile("t2star_exclude_fill.nii.gz"))

class T1Molli(Module):
    def __init__(self):
        Module.__init__(self, "t1_molli")

    def process(self):
        add_niftis = self.pipeline.options.add_niftis
        base_subjid = self.pipeline.options.subjid
        while 1:
            t1s = os.path.join(add_niftis, base_subjid)
            t1 = self.single_inimg("molli_t1_map_nifti", "*.nii.gz", src=t1s)
            if t1:
                break
            base_subjid = base_subjid[:base_subjid.rfind("_")]
            if not base_subjid:
                break
        if t1:
            LOG.info(f" - Saving MOLLI T1 map from {t1.fname}")
            map = t1.data[..., 0]
            conf = t1.data[..., 1]
            t1.save_derived(map, self.outfile("t1_map.nii.gz"))
            t1.save_derived(map, self.outfile("t1_conf.nii.gz"))

class T1SE(Module):
    def __init__(self):
        Module.__init__(self, "t1_se")

    def process(self):
        add_niftis = self.pipeline.options.add_niftis
        base_subjid = self.pipeline.options.subjid
        while 1:
            t1s = os.path.join(add_niftis, base_subjid)
            t1 = self.single_inimg("seepi_t1_map_nifti", "*.nii.gz", src=t1s)
            if t1:
                break
            base_subjid = base_subjid[:base_subjid.rfind("_")]
            if not base_subjid:
                break
        if t1:
            LOG.info(f" - Saving SE T1 map from {t1.fname}")
            t1.save(self.outfile("t1.nii.gz"))

class LiverMaskIntersection(Module):
    def __init__(self):
        Module.__init__(self, name="liver_mask_intersection")

    def process(self):
        liver_mask = self.single_inimg("seg_liver_dixon_fix", "liver.nii.gz", src=self.OUTPUT)
        if not liver_mask:
            self.no_data("No liver masks found")

        t1_se = self.single_inimg("t1_se", "t1.nii.gz", src=self.OUTPUT)
        if not t1_se:
            self.no_data("No SE T1 map found")

        t1_molli = self.single_inimg("t1_molli", "t1_conf.nii.gz", src=self.OUTPUT)
        if not t1_molli:
            self.no_data("No MOLLI T1 map found")

        LOG.info(f"Liver mask (native) has {np.count_nonzero(liver_mask.data)} voxels")
        liver_mask_t1se = self.resample(liver_mask, t1_se, is_roi=True, allow_rotated=True)
        liver_mask_t1se = t1_se.save_derived(liver_mask_t1se.get_fdata().astype(np.uint8), self.outfile("liver_mask_t1se.nii.gz"))
        LOG.info(f"Liver mask (T1SE space) has {np.count_nonzero(liver_mask_t1se.data)} voxels")
        liver_mask_molli = self.resample(liver_mask, t1_molli, is_roi=True, allow_rotated=True)
        liver_mask_molli = t1_molli.save_derived(liver_mask_molli.get_fdata().astype(np.uint8), self.outfile("liver_mask_molli.nii.gz"))
        LOG.info(f"Liver mask (MOLLI space) has {np.count_nonzero(liver_mask_molli.data)} voxels")
        liver_mask_molli_t1se = self.resample(liver_mask_molli, t1_se, is_roi=True, allow_rotated=True)
        liver_mask_molli_t1se = t1_se.save_derived(liver_mask_molli_t1se.get_fdata().astype(np.uint8), self.outfile("liver_mask_molli_t1se.nii.gz"))
        LOG.info(f"Liver mask (MOLLI space -> T1SE space) has {np.count_nonzero(liver_mask_molli_t1se.data)} voxels")
        liver_mask_t1se_molli = self.resample(liver_mask_t1se, t1_molli, is_roi=True, allow_rotated=True)
        liver_mask_t1se_molli = t1_molli.save_derived(liver_mask_t1se_molli.get_fdata().astype(np.uint8), self.outfile("liver_mask_t1se_molli.nii.gz"))
        LOG.info(f"Liver mask (T1SE space -> MOLLI space) has {np.count_nonzero(liver_mask_t1se_molli.data)} voxels")

        liver_mask_t1se_intersection = np.logical_and(liver_mask_t1se.data > 0, liver_mask_molli_t1se.data > 0).astype(np.uint8)
        self.lightbox(t1_se, liver_mask_t1se_intersection, self.outfile("liver_mask_t1se_intersection.png"))
        liver_mask_t1se.save_derived(liver_mask_t1se_intersection.astype(np.uint8), self.outfile("liver_mask_t1se_intersection.nii.gz"))
        LOG.info(f"Liver mask intersection (T1SE space) has {np.count_nonzero(liver_mask_t1se_intersection.data)} voxels")
        liver_mask_molli_intersection = np.logical_and(liver_mask_molli.data > 0, liver_mask_t1se_molli.data > 0).astype(np.uint8)
        self.lightbox(t1_molli, liver_mask_molli_intersection, self.outfile("liver_mask_molli_intersection.png"))
        liver_mask_molli.save_derived(liver_mask_molli_intersection.astype(np.uint8), self.outfile("liver_mask_molli_intersection.nii.gz"))
        LOG.info(f"Liver mask intersection (MOLLI space) has {np.count_nonzero(liver_mask_t1se_intersection.data)} voxels")


class Radiomics(statistics.Radiomics):
    def __init__(self):
        statistics.Radiomics.__init__(
            self,
            params={
                "t1_molli" : {"dir" : "t1_molli", "fname" : "t1_conf.nii.gz", "maxval" : 1250},
                "t1_se" : {"dir" : "t1_se", "fname" : "t1.nii.gz", "maxval" : 1250},
            },
            segs = {
                "liver" : {"dir" : "seg_liver_dixon_fix", "fname" : "liver.nii.gz"},
            }
        )

class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats",
            default_limits="3t",
            segs={
                "liver_intersection_t1se" : {
                    "dir" : "liver_mask_intersection",
                    "glob" : "liver_mask_t1se_intersection.nii.gz"
                },
                "liver_intersection_molli" : {
                    "dir" : "liver_mask_intersection",
                    "glob" : "liver_mask_molli_intersection.nii.gz"
                },
            },
            params={
                "t2star" : {
                    "dir" : "t2star",
                    "glob" : "t2star_exclude_fill.nii.gz",
                    "segs" : ["liver_intersection_molli"],
                },
                "ff" : {
                    "dir" : "dixon",
                    "src" : self.INPUT,
                    "glob" : "fat_fraction.nii.gz",
                    "segs" : ["liver_intersection_molli"],
                },
                "t1_molli" : {
                    "dir" : "t1_molli",
                    "glob" : "t1_conf.nii.gz",
                    "segs" : ["liver_intersection_molli"],
                },
                "t1_se" : {
                    "dir" : "t1_se",
                    "glob" : "t1.nii.gz",
                    "segs" : ["liver_intersection_t1se"],
                },
            },
            stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

__version__ = "0.0.1"

NAME = "mrquee_bsmart_uon"

MODULES = [
    # Segmentations
    segmentations.LiverDixon(),
    # Parameter maps
    T2Star(),
    T1Molli(),
    T1SE(),
    # Post-processing of segmentations
    seg_postprocess.SegFix(
        "seg_liver_dixon",
        fix_dir_option="liver_masks",
        segs={
            "liver.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="../dixon",
        map_fname="water.nii.gz"
    ),
    LiverMaskIntersection(),
    # Statistics
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--add-niftis", help="Dir containing additional NIFTI maps")
    parser.add_argument("--liver-masks", help="Directory containing manual liver masks")
