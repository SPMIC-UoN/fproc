import logging

import numpy as np

from fproc.module import Module, CopyModule
from fproc.modules import segmentations, seg_postprocess, statistics, maps

LOG = logging.getLogger(__name__)

class PancreasSegRestricted(Module):
    def __init__(self):
        Module.__init__(self, "seg_pancreas_ethrive_restricted")

    def process(self):
        seg_orig = self.inimg("seg_pancreas_ethrive_fix_largestblob", "pancreas.nii.gz", is_depfile=True)
        ff = self.inimg("fat_fraction", "fat_fraction.nii.gz", is_depfile=True)
        ff_resamp = self.resample(ff, seg_orig, is_roi=False).get_fdata().squeeze()
        ff_30 = ff_resamp < 30
        ff_50 = ff_resamp < 50
        seg_30 = np.logical_and(seg_orig.data > 0, ff_30)
        seg_50 = np.logical_and(seg_orig.data > 0, ff_50)
        LOG.info(" - Saving pancreas masks restricted by fat fraction")
        seg_orig.save_derived(ff_resamp, self.outfile("fat_fraction.nii.gz"))
        seg_orig.save_derived(ff_30, self.outfile("fat_fraction_lt_30.nii.gz"))
        seg_orig.save_derived(ff_50, self.outfile("fat_fraction_lt_50.nii.gz"))
        seg_orig.save_derived(seg_30, self.outfile("seg_pancreas_ff_lt_30.nii.gz"))
        seg_orig.save_derived(seg_50, self.outfile("seg_pancreas_ff_lt_50.nii.gz"))

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
                "fat_fraction" : {"dir" : "fat_fraction", "fname" : "fat_fraction_scanner.nii.gz"},
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
                "sat" : {
                    "dir" : "seg_sat_dixon",
                    "glob" : "sat.nii.gz",
                    "params" : [],
                },
                "vat" : {
                    "dir" : "seg_vat_dixon",
                    "glob" : "vat.nii.gz",
                    "params" : [],
                },
                "kidney_dixon" : {
                    "dir" : "seg_kidney_dixon",
                    "glob" : "kidney.nii.gz"
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
                    "glob" : "fat_fraction_scanner.nii.gz",
                    "limits" : (0, 100),
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
    segmentations.BodyDixon(),
    segmentations.SatDixon(),
    segmentations.LiverDixon(),
    segmentations.SpleenDixon(),
    segmentations.KidneyDixon(model_id="422"),
    segmentations.PancreasEthrive(),
    segmentations.KidneyT2w(),
    segmentations.KidneyT1(),
    # Parameter maps
    maps.FatFractionDixon(),
    maps.T2starDixon(),
    CopyModule("b0_ax", in_name="b0"),
    CopyModule("b0_cor", in_name="b0"),
    CopyModule("b1_ax", in_name="b1"),
    CopyModule("b1_cor", in_name="b1"),
    T1Kidney(),
    # Post-processing of segmentations
    seg_postprocess.KidneyT1Clean(),
    seg_postprocess.SegFix(
        "seg_pancreas_ethrive",
        fix_dir_option="pancreas_masks",
        segs={
            "pancreas.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="dixon",
        map_fname="water.nii.gz"
    ),
    seg_postprocess.SegFix(
        "seg_liver_dixon",
        fix_dir_option="liver_masks",
        segs={
            "liver.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="dixon",
        map_fname="water.nii.gz"
    ),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive_fix", "pancreas.nii.gz"),
    PancreasSegRestricted(),
    segmentations.VatDixon(
        organs={
            "seg_liver_dixon_fix" : "liver.nii.gz",
            "seg_spleen_dixon" : "spleen.nii.gz",
            "seg_pancreas_ethrive_fix_largestblob" : "pancreas.nii.gz",
            "seg_kidney_t2w" : "kidney_mask.nii.gz"
        }
    ),
    # Statistics
    Radiomics(),
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--pancreas-masks", help="Directory containing manual pancreas masks")
    parser.add_argument("--liver-masks", help="Directory containing manual liver masks")
