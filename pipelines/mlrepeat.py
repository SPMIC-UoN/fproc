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
        ff = self.inimg("fat_fraction", "fat_fraction_scanner.nii.gz", is_depfile=True)
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
                "t2star" : {"dir" : "t2star_dixon", "fname" : "t2star_exclude_fill.nii.gz", "minval" : 0},
                "fat_fraction" : {"dir" : "fat_fraction", "fname" : "fat_fraction_scanner.nii.gz"},
                "t1" : {"dir" : "t1_kidney", "fname" : "t1_map.nii.gz"},
            },
            segs = {
                "pancreas" : {"dir" : "seg_pancreas_ethrive_fix", "fname" : "pancreas.nii.gz"},
                "spleen" : {"dir" : "seg_spleen_dixon_102", "fname" : "spleen.nii.gz"},
                "kidney" : {"dir" : "seg_kidney_t2w", "fname" : "kidney_mask.nii.gz"},
            }
        )

class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats", 
            segs={
                "liver_14" : {
                    "dir" : "seg_liver_dixon_14",
                    "glob" : "liver.nii.gz"
                },
                "liver_452" : {
                    "dir" : "seg_liver_dixon_452",
                    "glob" : "liver.nii.gz"
                },
                "liver_423" : {
                    "dir" : "seg_liver_dixon_423",
                    "glob" : "liver.nii.gz"
                },
                "liver_427" : {
                    "dir" : "seg_liver_dixon_427",
                    "glob" : "liver.nii.gz"
                },
                "spleen_102" : {
                    "dir" : "seg_spleen_dixon_102",
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
                "kidney_dixon_422" : {
                    "dir" : "seg_kidney_dixon_422",
                    "glob" : "kidney.nii.gz"
                },
                "kidney_dixon_left_422" : {
                    "dir" : "seg_kidney_dixon_422",
                    "glob" : "kidney_left.nii.gz"
                },
                "kidney_dixon_right_422" : {
                    "dir" : "seg_kidney_dixon_422",
                    "glob" : "kidney_right.nii.gz"
                },
                "kidney_dixon_426" : {
                    "dir" : "seg_kidney_dixon_426",
                    "glob" : "kidney.nii.gz"
                },
                "kidney_dixon_left_426" : {
                    "dir" : "seg_kidney_dixon_426",
                    "glob" : "kidney_left.nii.gz"
                },
                "kidney_dixon_right_426" : {
                    "dir" : "seg_kidney_dixon_426",
                    "glob" : "kidney_right.nii.gz"
                },
                "kidney_dixon_450" : {
                    "dir" : "seg_kidney_dixon_450",
                    "glob" : "kidney.nii.gz"
                },
                "kidney_dixon_left_450" : {
                    "dir" : "seg_kidney_dixon_450",
                    "glob" : "kidney_left.nii.gz"
                },
                "kidney_dixon_right_450" : {
                    "dir" : "seg_kidney_dixon_450",
                    "glob" : "kidney_right.nii.gz"
                },
                "kidney_dixon_nofat" : {
                    "dir" : "seg_kidney_fat_dixon",
                    "glob" : "kidney_parenchyma.nii.gz",
                    "params" : [],
                },
                "kidney_dixon_nofat_left" : {
                    "dir" : "seg_kidney_fat_dixon",
                    "glob" : "kidney_parenchyma_left.nii.gz",
                    "params" : [],
                },
                "kidney_dixon_nofat_right" : {
                    "dir" : "seg_kidney_fat_dixon",
                    "glob" : "kidney_parenchyma_right.nii.gz",
                    "params" : [],
                },
                "fat_pelvis" : {
                    "dir" : "seg_kidney_fat_dixon",
                    "glob" : "fat_pelvis.nii.gz",
                    "params" : ["ff"],
                },
                "fat_pelvis_left" : {
                    "dir" : "seg_kidney_fat_dixon",
                    "glob" : "fat_pelvis_left.nii.gz",
                    "params" : ["ff"],
                },
                "fat_pelvis_right" : {
                    "dir" : "seg_kidney_fat_dixon",
                    "glob" : "fat_pelvis_right.nii.gz",
                    "params" : ["ff"],
                },
            },
            params={
                "t2star" : {
                    "dir" : "t2star_dixon",
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
    # Parameter maps
    maps.DixonDerived(),
    maps.FatFractionDixon(),
    maps.T2starDixon(),
    CopyModule("b0_ax", in_name="b0"),
    CopyModule("b0_cor", in_name="b0"),
    CopyModule("b1_ax", in_name="b1"),
    CopyModule("b1_cor", in_name="b1"),
    T1Kidney(),

    # Segmentations
    segmentations.BodyDixon(),
    segmentations.SatDixon(),
    segmentations.OrganDixon(organ="liver", dixon_dir="../fproc/dixon", name="seg_liver_dixon_14", model_id="14", inputs=["fat", "t2star", "water"]),
    segmentations.OrganDixon(organ="liver", dixon_dir="../fproc/dixon", name="seg_liver_dixon_452", model_id="452", inputs=["fat", "op", "water"]),
    segmentations.OrganDixon(organ="liver", dixon_dir="../fproc/dixon", name="seg_liver_dixon_423", model_id="423", inputs=["fat", "water"]),
    segmentations.OrganDixon(organ="liver", dixon_dir="../fproc/dixon", name="seg_liver_dixon_427", model_id="427", inputs=["ip", "op"]),
    segmentations.OrganDixon(organ="spleen", dixon_dir="../fproc/dixon", name="seg_spleen_dixon_102", model_id="102", inputs=["fat", "t2star", "water"]),
    segmentations.OrganDixon(organ="kidney", dixon_dir="../fproc/dixon", name="seg_kidney_dixon_422", model_id="422", inputs=["fat", "fat_fraction", "t2star", "water"]),
    segmentations.OrganDixon(organ="kidney", dixon_dir="../fproc/dixon", name="seg_kidney_dixon_426", model_id="426", inputs=["fat", "water"]),
    segmentations.OrganDixon(organ="kidney", dixon_dir="../fproc/dixon", name="seg_kidney_dixon_450", model_id="450", inputs=["fat", "t2star", "water"]),
    segmentations.PancreasEthrive(),
    segmentations.KidneyT2w(),
    segmentations.KidneyT1(map_dir="../fsort/t1_molli"),
    segmentations.TotalSeg(
        src_dir="../fsort/dixon",
        img_glob="water.nii.gz"
    ),

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
        "seg_liver_dixon_14",
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
        ff_glob="fat_fraction_scanner.nii.gz",
        organs={
            "seg_liver_dixon_14" : "liver.nii.gz",
            "seg_spleen_dixon_102" : "spleen.nii.gz",
            "seg_pancreas_ethrive_fix_largestblob" : "pancreas.nii.gz",
            "seg_kidney_t2w" : "kidney_mask.nii.gz"
        }
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon",
        ff_dir="fat_fraction",
        ff_glob="fat_fraction_scanner.nii.gz",
        kidney_seg_dir="seg_kidney_dixon_422",
        kidney_seg_glob="kidney.nii.gz",
        ff_thresh=15,
    ),
    # Statistics
    Radiomics(),
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--pancreas-masks", help="Directory containing manual pancreas masks")
    parser.add_argument("--liver-masks", help="Directory containing manual liver masks")
