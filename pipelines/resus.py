import logging
import os

import numpy as np

from fproc.module import Module
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

class T1Molli(Module):
    def __init__(self):
        Module.__init__(self, "t1_molli")

    def process(self):
        add_niftis = self.pipeline.options.add_niftis
        base_subjid = self.pipeline.options.subjid
        t1s = os.path.join(add_niftis, base_subjid + "*")
        t1 = self.single_inimg("molli_t1_map_nifti", "*.nii.gz", src=t1s)
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
        t1s = os.path.join(add_niftis, base_subjid + "*")
        t1 = self.single_inimg("seepi_t1_map_nifti", "*.nii.gz", src=t1s)
        if t1:
            LOG.info(f" - Saving SE T1 map from {t1.fname}")
            t1.save(self.outfile("t1.nii.gz"))

class Radiomics(statistics.Radiomics):
    def __init__(self):
        statistics.Radiomics.__init__(
            self,
            params={
                "t1_molli" : {"dir" : "t1_molli", "fname" : "t1_conf.nii.gz", "maxval" : 1400},
                "t1_se" : {"dir" : "t1_se", "fname" : "t1.nii.gz", "maxval" : 1400},
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
                "liver" : {
                    "dir" : "seg_liver_dixon_fix",
                    "glob" : "liver.nii.gz"
                },
                "spleen" : {
                    "dir" : "seg_spleen_dixon",
                    "glob" : "spleen.nii.gz"
                },
                "kidney_dixon" : {
                    "dir" : "seg_kidney_dixon",
                    "glob" : "kidney.nii.gz"
                },
                "pancreas" : {
                    "dir" : "seg_pancreas_ethrive_fix",
                    "glob" : "pancreas.nii.gz",
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
                "kidney_t2w" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "kidney_mask.nii.gz",
                },
            },
            params={
                "t2star" : {
                    "dir" : "t2star",
                    "glob" : "t2star_exclude_fill.nii.gz",
                },
                "ff" : {
                    "dir" : "fat_fraction",
                    "glob" : "fat_fraction_scanner.nii.gz",
                },
                "t1_molli" : {
                    "dir" : "t1_molli",
                    "glob" : "t1_conf.nii.gz",
                },
                "t1_se" : {
                    "dir" : "t1_se",
                    "glob" : "t1.nii.gz",
                },
            },
            stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

__version__ = "0.0.1"

NAME = "resus"

MODULES = [
    # Segmentations
    segmentations.BodyDixon(),
    segmentations.SatDixon(),
    segmentations.LiverDixon(),
    segmentations.SpleenDixon(),
    segmentations.KidneyDixon(model_id="422"),
    segmentations.PancreasEthrive(),
    segmentations.KidneyT2w(),
    # Parameter maps
    maps.FatFractionDixon(),
    maps.T2starDixon(),
    T1Molli(),
    T1SE(),
    # Post-processing of segmentations
    seg_postprocess.SegFix(
        "seg_pancreas_ethrive",
        fix_dir_option="pancreas_masks",
        segs={
            "pancreas.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="../dixon",
        map_fname="water.nii.gz"
    ),
    seg_postprocess.SegFix(
        "seg_liver_dixon",
        fix_dir_option="liver_masks",
        segs={
            "liver.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="../dixon",
        map_fname="water.nii.gz"
    ),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive_fix", "pancreas.nii.gz"),
    PancreasSegRestricted(),
    segmentations.VatDixon(
        ff_glob="fat_fraction_scanner.nii.gz",
        organs={
            "seg_liver_dixon_fix" : "liver.nii.gz",
            "seg_spleen_dixon" : "spleen.nii.gz",
            "seg_pancreas_ethrive_fix_largestblob" : "pancreas.nii.gz",
            "seg_kidney_dixon" : "kidney.nii.gz"
        }
    ),
    # Statistics
    Radiomics(),
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--add-niftis", help="Dir containing additional NIFTI maps")
    parser.add_argument("--pancreas-masks", help="Directory containing manual pancreas masks")
    parser.add_argument("--liver-masks", help="Directory containing manual liver masks")
