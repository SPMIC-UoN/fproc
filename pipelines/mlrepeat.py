import logging

import numpy as np

from fproc.module import Module, CopyModule
from fproc.modules import segmentations, seg_postprocess, statistics

LOG = logging.getLogger(__name__)

class PancreasSegRestricted(Module):
    def __init__(self):
        Module.__init__(self, "seg_pancreas_ethrive_restricted")

    def process(self):
        seg_orig = self.inimg("seg_pancreas_ethrive_fix_largestblob", "pancreas.nii.gz", is_depfile=True)
        ff = self.inimg("fat_fraction", "fat_fraction.nii.gz", is_depfile=True)
        ff_resamp = self.resample(ff, seg_orig, is_roi=False).get_fdata().squeeze()
        ff_30 = ff_resamp < 0.3
        ff_50 = ff_resamp < 0.5
        seg_30 = np.logical_and(seg_orig.data > 0, ff_30)
        seg_50 = np.logical_and(seg_orig.data > 0, ff_50)
        LOG.info(" - Saving pancreas masks restricted by fat fraction")
        seg_orig.save_derived(ff_resamp, self.outfile("fat_fraction.nii.gz"))
        seg_orig.save_derived(ff_30, self.outfile("fat_fraction_lt_30.nii.gz"))
        seg_orig.save_derived(ff_50, self.outfile("fat_fraction_lt_50.nii.gz"))
        seg_orig.save_derived(seg_30, self.outfile("seg_pancreas_ff_lt_30.nii.gz"))
        seg_orig.save_derived(seg_50, self.outfile("seg_pancreas_ff_lt_50.nii.gz"))

class FatFraction(Module):
    def __init__(self):
        Module.__init__(self, "fat_fraction")

    def process(self):
        fat = self.inimg("dixon", "fat.nii.gz")
        water = self.inimg("dixon", "water.nii.gz")
        ff_orig = self.inimg("dixon", "fat_fraction.nii.gz")
        ff_orig.save(self.outfile("fat_fraction_orig.nii.gz"))

        ff = fat.data.astype(np.float32) / (fat.data + water.data)
        fat.save_derived(ff, self.outfile("fat_fraction.nii.gz"))

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

class T1Kidney(Module):
    def __init__(self):
        Module.__init__(self, "t1_kidney")

    def process(self):
        t1_map = self.inimg("molli_kidney", "t1_map.nii.gz")
        t1_map.save(self.outfile("t1_map.nii.gz"))
        t1_conf = self.inimg("molli_kidney", "t1_conf.nii.gz")
        t1_conf.save(self.outfile("t1_conf.nii.gz"))

class Radiomics(statistics.Radiomics):
    def __init__(self):
        statistics.Radiomics.__init__(
            self,
            params={
                "t2star" : {"dir" : "t2star", "fname" : "t2star_exclude_fill.nii.gz", "minval" : 0},
                "fat_fraction" : {"dir" : "fat_fraction", "fname" : "fat_fraction.nii.gz"},
                "t1" : {"dir" : "t1_kidney", "fname" : "t1_map.nii.gz"},
            },
            segs = {
                "pancreas" : {"dir" : "seg_pancreas_ethrive_fix", "fname" : "pancreas.nii.gz"},
                "spleen" : {"dir" : "seg_spleen_dixon", "fname" : "spleen.nii.gz"},
                "kidney" : {"dir" : "seg_kidney_t2w", "fname" : "kidney_mask.nii.gz"},
            }
        )

class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats", 
            segs={
                "liver" : {
                    "dir" : "seg_liver_dixon_fix",
                    "glob" : "liver.nii.gz"
                },
                "spleen" : {
                    "dir" : "seg_spleen_dixon",
                    "glob" : "spleen.nii.gz"
                },
                "pancreas" : {
                    "dir" : "seg_pancreas_ethrive_fix",
                    "glob" : "pancreas.nii.gz",
                },
                "kidney_cortex_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "kidney_cortex_l_t1.nii.gz"
                },
                "kidney_cortex_r" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "kidney_cortex_r_t1.nii.gz"
                },
                "kidney_medulla_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "kidney_medulla_l_t1.nii.gz"
                },
                "kidney_medulla_r" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "kidney_medulla_r_t1.nii.gz"
                },
                "kidney_left" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "kidney_left_kidney.nii.gz"
                },
                "kidney_right" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "kidney_right_kidney.nii.gz"
                },
            },
            params={
                "t2star" : {
                    "dir" : "t2star",
                    "glob" : "t2star_exclude_fill.nii.gz",
                    "limits" : (2, 100),
                },
                "ff" : {
                    "dir" : "fat_fraction",
                    "glob" : "fat_fraction.nii.gz",
                    "limits" : (0, 1),
                },
                "b0_ax" : {
                    "dir" : "b0_ax",
                    "glob" : "b0.nii.gz",
                },
                "b1_ax" : {
                    "dir" : "b1_ax",
                    "glob" : "b1.nii.gz",
                },
                "b0_cor" : {
                    "dir" : "b0_cor",
                    "glob" : "b0.nii.gz",
                },
                "b1_cor" : {
                    "dir" : "b1_cor",
                    "glob" : "b1.nii.gz",
                },
                "t1_kidney" : {
                    "dir" : "t1_kidney",
                    "glob" : "t1_map.nii.gz",
                },
            },
            stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

__version__ = "0.0.1"

NAME = "ML_repeat"

MODULES = [
    # Segmentations
    segmentations.LiverDixon(),
    segmentations.SpleenDixon(),
    segmentations.PancreasEthrive(),
    segmentations.KidneyT2w(),
    segmentations.KidneyT1(),
    # Parameter maps
    FatFraction(),
    T2Star(),
    CopyModule("b0_ax", in_name="b0"),
    CopyModule("b0_cor", in_name="b0"),
    CopyModule("b1_ax", in_name="b1"),
    CopyModule("b1_cor", in_name="b1"),
    T1Kidney(),
    # Post-processing of segmentations
    seg_postprocess.KidneyT1Clean(),
    seg_postprocess.SegFix("seg_pancreas_ethrive", "pancreas.nii.gz", "pancreas_masks", "dixon", "water.nii.gz"),
    seg_postprocess.SegFix("seg_liver_dixon", "liver.nii.gz", "liver_masks", "dixon", "water.nii.gz"),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive_fix", "pancreas.nii.gz"),
    PancreasSegRestricted(),
    # Statistics
    Radiomics(),
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--kidney-t1-model", help="Filename or URL for T1 segmentation model weights", default="/spmstore/project/RenalMRI/trained_models/kidney_t1_molli_min_max.pt")
    parser.add_argument("--kidney-t2w-model", help="Filename or URL for T2w segmentation model weights")
    parser.add_argument("--pancreas-masks", help="Directory containing manual pancreas masks")
    parser.add_argument("--liver-masks", help="Directory containing manual liver masks")
