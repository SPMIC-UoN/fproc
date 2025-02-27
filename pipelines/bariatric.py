import glob
import logging
import os

import numpy as np
import scipy

from fsort import ImageFile
from fproc.module import Module, CopyModule
from fproc.modules import maps, statistics, segmentations, seg_postprocess

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class PancreasSegEro(Module):
    def __init__(self):
        Module.__init__(self, "seg_pancreas_ethrive_ero")

    def process(self):
        seg_origs = self.inimgs("seg_pancreas_ethrive_fix", "pancreas.nii.gz", is_depfile=True)
        if not seg_origs:
            self.no_data("No pancreas segementations found")
        else:
            if len(seg_origs) > 1:
                LOG.warn(" - Multiple pancreas segmentations found - using first")
            seg_orig = seg_origs[0]
            ero_data = scipy.ndimage.morphology.binary_erosion(seg_orig.data, iterations=2)
            seg_ero = seg_orig.save_derived(ero_data, self.outfile("pancreas.nii.gz"))

            # Overlay onto dixon
            dixon_water = self.inimgs("dixon", "water.nii.gz")
            if not dixon_water:
                LOG.warn(f" - Could not find Dixon water image for overlay")
            else:
                self.lightbox(dixon_water[0], seg_ero, name="pancreas_ero_lightbox", tight=True)

class PancreasSegRestricted(Module):
    def __init__(self):
        Module.__init__(self, "seg_pancreas_ethrive_restrict")

    def process(self):
        seg_orig = self.inimg("seg_pancreas_ethrive_fix", "pancreas.nii.gz", is_depfile=True)
        ff = self.inimg("fat_fraction", "fat_fraction.nii.gz", is_depfile=True)
        ff_resamp = self.resample(ff, seg_orig, is_roi=False).get_fdata().squeeze()
        ff_30 = ff_resamp < 30
        ff_50 = ff_resamp < 50
        seg_30 = np.logical_and(seg_orig.data > 0, ff_30)
        seg_50 = np.logical_and(seg_orig.data > 0, ff_50)
        seg_orig.save_derived(ff_resamp, self.outfile("fat_fraction.nii.gz"))
        seg_orig.save_derived(ff_30, self.outfile("fat_fraction_lt_30.nii.gz"))
        seg_orig.save_derived(ff_50, self.outfile("fat_fraction_lt_50.nii.gz"))
        seg_orig.save_derived(seg_30, self.outfile("pancreas_ff_lt_30.nii.gz"))
        seg_orig.save_derived(seg_50, self.outfile("pancreas_ff_lt_50.nii.gz"))

class T2(CopyModule):
    def __init__(self):
        CopyModule.__init__(self, "t2")

class T1Liver(Module):
    def __init__(self):
        Module.__init__(self, "t1_liver")

    def process(self):
        t1_map = self.inimg("molli_liver", "t1_map.nii.gz")
        t1_map.save(self.outfile("t1_map.nii.gz"))

class T1Kidney(Module):
    def __init__(self):
        Module.__init__(self, "t1_kidney")

    def process(self):
        t1_map = self.inimg("molli_kidney", "t1_map.nii.gz")
        t1_map.save(self.outfile("t1_map.nii.gz"))
        t1_conf = self.inimg("molli_kidney", "t1_conf.nii.gz")
        t1_conf.save(self.outfile("t1_conf.nii.gz"))

class SeT1Map(Module):
    def __init__(self):
        Module.__init__(self, "se_t1")

    def process(self):
        if not self.pipeline.options.se_t1_maps:
            self.no_data("No path to additional SE T1 maps given")
        else:
            globexpr = os.path.join(
                self.pipeline.options.se_t1_maps, 
                "%s_*.nii.gz" % self.pipeline.options.subjid
            )
            maps = glob.glob(globexpr)
            if not maps:
                LOG.info(f" - No SE T1 maps for {self.pipeline.options.subjid} in {globexpr}")
            else:
                if len(maps) > 1:
                    LOG.warn(f" - Multiple SE T1 maps found for {self.pipeline.options.subjid}: {maps} - using first")
                map = ImageFile(maps[0])
                map.save(self.outfile("se_t1.nii.gz"))

class AdcMap(Module):
    def __init__(self):
        Module.__init__(self, "adc")

    def process(self):
        if not self.pipeline.options.adc_maps:
            self.no_data("No path to additional ADC maps given")
        else:
            globexpr = os.path.join(
                self.pipeline.options.adc_maps, 
                "%s_*.nii.gz" % self.pipeline.options.subjid
            )
            maps = glob.glob(globexpr)
            if not maps:
                LOG.info(f" - No ADC maps for {self.pipeline.options.subjid} in {globexpr}")
            else:
                if len(maps) > 1:
                    LOG.warn(f" - Multiple ADC maps found for {self.pipeline.options.subjid}: {maps} - using first")
                map = ImageFile(maps[0])
                map.save(self.outfile("adc.nii.gz"))

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
                "pancreas_nonero" : {
                    "dir" : "seg_pancreas_ethrive_fix",
                    "glob" : "pancreas.nii.gz",
                    "params" : []  # Volumes only
                },
                "pancreas" : {
                    # Eroded pancreas seg for params
                    "dir" : "seg_pancreas_ethrive_ero",
                    "glob" : "pancreas.nii.gz",
                },
                "pancreas_ff_lt_30" : {
                    "dir" : "seg_pancreas_ethrive_restrict",
                    "glob" : "pancreas_ff_lt_30.nii.gz",
                },
                "pancreas_ff_lt_50" : {
                    "dir" : "seg_pancreas_ethrive_restrict",
                    "glob" : "pancreas_ff_lt_50.nii.gz",
                },
                "sat" : {
                    "dir" : "seg_sat_dixon_fix",
                    "glob" : "sat.nii.gz",
                    "params" : []  # Volumes only
                },
                "vat" : {
                    "dir" : "seg_vat_dixon",
                    "glob" : "vat.nii.gz",
                    "params" : []  # Volumes only
                },
                "kidney_cortex_l" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "kidney_cortex_l_t1.nii.gz"
                },
                "kidney_cortex_r" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "kidney_cortex_r_t1.nii.gz"
                },
                "kidney_medulla_l" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "kidney_medulla_l_t1.nii.gz"
                },
                "kidney_medulla_r" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "kidney_medulla_r_t1.nii.gz"
                },
                "kidney_dixon" : {
                    "dir" : "seg_kidney_dixon",
                    "glob" : "kidney.nii.gz",
                    "params" : []  # Volumes only
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
                "t1_liver" : {
                    "dir" : "t1_liver",
                    "glob" : "t1_map.nii.gz",
                },
                "t1_kidney" : {
                    "dir" : "t1_kidney",
                    "glob" : "t1_map.nii.gz",
                },
                "b0" : {
                    "dir" : "b0",
                    "glob" : "b0.nii.gz",
                },
                "b1" : {
                    "dir" : "b1",
                    "glob" : "b1.nii.gz",
                },
                "t2" : {
                    "dir" : "t2",
                    "glob" : "t2.nii.gz",
                },
                "se_t1" : {
                    "dir" : "se_t1",
                    "glob" : "se_t1.nii.gz",
                },
                "adc" : {
                    "dir" : "adc",
                    "glob" : "adc.nii.gz",
                },
            },
            stats=["iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

NAME="bariatric"

MODULES = [
    maps.FatFractionDixon(),
    maps.B0(),
    CopyModule("b1"),
    T2(),
    maps.T2starDixon(),
    T1Liver(),
    T1Kidney(),
    SeT1Map(),
    AdcMap(),
    segmentations.KidneyDixon(model_id="422"),
    segmentations.LiverDixon(),
    segmentations.SpleenDixon(),
    segmentations.SatDixon(),
    segmentations.BodyDixon(),
    segmentations.KidneyT1(),
    segmentations.PancreasEthrive(),
    seg_postprocess.KidneyT1Clean(t2w=False),
    seg_postprocess.SegFix(
        "seg_pancreas_ethrive",
        fix_dir_option="pancreas_masks",
        segs={
            "pancreas.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="dixon",
        map_fname="water.nii.gz",
        map_src=Module.INPUT,
    ),
    seg_postprocess.SegFix(
        "seg_liver_dixon",
        fix_dir_option="liver_masks",
        segs={
            "liver.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="dixon",
        map_fname="water.nii.gz",
        map_src=Module.INPUT,
    ),
    seg_postprocess.SegFix(
        "seg_sat_dixon",
        fix_dir_option="sat_masks",
        segs={
            "sat.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="dixon",
        map_fname="water.nii.gz",
        map_src=Module.INPUT,
    ),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive_fix", "pancreas.nii.gz"),
    PancreasSegEro(),
    PancreasSegRestricted(),
    segmentations.VatDixon(
        organs={
            "seg_liver_dixon_fix" : "liver.nii.gz",
            "seg_spleen_dixon" : "spleen.nii.gz",
            "seg_pancreas_ethrive_fix_largestblob" : "pancreas.nii.gz",
            "seg_kidney_dixon" : "kidney.nii.gz"
        }
    ),
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--pancreas-masks", help="Directory containing manual pancreas masks")
    parser.add_argument("--liver-masks", help="Directory containing manual liver masks")
    parser.add_argument("--sat-masks", help="Directory containing manual SAT masks")
    parser.add_argument("--kidney-masks", help="Directory containing manual kidney cortex/medulla masks")
    parser.add_argument("--se-t1-maps", help="Directory containing additional SE T1 maps")
    parser.add_argument("--adc-maps", help="Directory containing additional ADC maps")
