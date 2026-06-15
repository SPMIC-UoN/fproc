import logging

import numpy as np

from fproc.module import Module, CopyModule
from fproc.modules import segmentations, seg_postprocess, statistics, maps

LOG = logging.getLogger(__name__)

class T1Kidney(Module):
    def __init__(self):
        Module.__init__(self, "t1_kidney")

    def process(self):
        t1_map = self.inimg("molli_kidney", "t1_map.nii.gz")
        t1_map.save(self.outfile("t1_map.nii.gz"))
        t1_conf = self.inimg("molli_kidney", "t1_conf.nii.gz")
        t1_conf.save(self.outfile("t1_conf.nii.gz"))

class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats", 
            segs={
                "liver" : {
                    "dir" : "totalseg",
                    "glob" : "liver.nii.gz"
                },
                "spleen" : {
                    "dir" : "totalseg",
                    "glob" : "spleen.nii.gz"
                },
                "pancreas" : {
                    "dir" : "totalseg",
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
                "tkv_left" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "kidney_left_kidney.nii.gz"
                },
                "tkv_right" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "kidney_right_kidney.nii.gz"
                },
                "sat" : {
                    "dir" : "totalseg",
                    "glob" : "subcutaneous_fat.nii.gz",
                    "params" : [],
                },
                "vat" : {
                    "dir" : "seg_vat_dixon",
                    "glob" : "vat.nii.gz",
                    "params" : [],
                },
                "kidney_dixon" : {
                    "dir" : "totalseg",
                    "glob" : "kidneys.nii.gz"
                },
                "kidney_dixon_left" : {
                    "dir" : "totalseg",
                    "glob" : "kidney_left.nii.gz"
                },
                "kidney_dixon_right" : {
                    "dir" : "totalseg",
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
                "r2star" : {
                    "dir" : "t2star_dixon",
                    "glob" : "r2star_t2star_exclude_fill.nii.gz",
                    "limits" : (10, 500),
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
            stats=["n", "iqn", "iqmean", "median", "iqstd", "perc90", "te", "mode", "fwhm", ],
            seg_volumes=True,
        )

__version__ = "0.0.1"

NAME = "ML_repeat"

MODULES = [
    ## Parameter maps

    maps.DixonDerived(),
    maps.FatFractionDixon(),
    maps.T2starDixon(),
    CopyModule("b0_ax", in_name="b0"),
    CopyModule("b0_cor", in_name="b0"),
    CopyModule("b1_ax", in_name="b1"),
    CopyModule("b1_cor", in_name="b1"),
    T1Kidney(),

    ## Segmentations

    segmentations.BodyDixon(),
    segmentations.KidneyT2w(),
    segmentations.KidneyT1(map_dir="../fsort/t1_molli"),
    seg_postprocess.KidneyT1Clean(),
    segmentations.TotalSeg(
        src_dir="../fsort/dixon",
        img_glob="water.nii.gz",
        dilate=1,
    ),

    segmentations.VatDixon(
        name="seg_vat_dixon",
        ff_dir="fat_fraction",
        ff_glob="fat_fraction_scanner.nii.gz",
        body_dir="seg_body_dixon",
        sat_dir="totalseg",
        sat_glob="subcutaneous_fat.nii.gz",
        organs={
            "totalseg": "liver.nii.gz",
            "totalseg": "spleen.nii.gz",
            "totalseg": "pancreas.nii.gz",
            "totalseg": "kidneys.nii.gz",
        },
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon",
        ff_dir="fat_fraction",
        ff_glob="fat_fraction_scanner.nii.gz",
        kidney_seg_dir="totalseg",
        kidney_seg_glob="kidneys.nii.gz",
        ff_thresh=15,
    ),

    # Statistics
    statistics.Radiomics(
        params={
            "t2star" : {"dir" : "t2star_dixon", "fname" : "t2star_exclude_fill.nii.gz", "minval" : 0},
            "fat_fraction" : {"dir" : "fat_fraction", "fname" : "fat_fraction_scanner.nii.gz"},
            "t1" : {"dir" : "t1_kidney", "fname" : "t1_map.nii.gz"},
        },
        segs = {
            "pancreas" : {"dir" : "totalseg", "fname" : "pancreas.nii.gz"},
            "spleen" : {"dir" : "totalseg", "fname" : "spleen.nii.gz"},
            "kidney" : {"dir" : "totalseg", "fname" : "kidneys.nii.gz"},
        }
    ),
    statistics.Radiomics(
        name="shape_metrics",
        deps=["totalseg", "fat_fraction"],
        params={
            "ff": {"dir": "fat_fraction", "fname": "fat_fraction_scanner.nii.gz"},
        },
        segs={
            "liver" : {"dir" : "totalseg", "fname" : "liver.nii.gz"},
            "pancreas" : {"dir" : "totalseg", "fname" : "pancreas.nii.gz"},
            "spleen" : {"dir" : "totalseg", "fname" : "spleen.nii.gz"},
            "kidney_left" : {"dir" : "totalseg", "fname" : "kidney_left.nii.gz"},
            "kidney_right" : {"dir" : "totalseg", "fname" : "kidney_right.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),

    SegStats(),
]
