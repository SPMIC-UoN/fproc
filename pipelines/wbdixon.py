"""
Whole-body dixon leg segmentation pipeline

Based on pre-stitched data from whole-body dixon
"""

import logging

from fproc.modules import maps, statistics, segmentations, seg_postprocess

LOG = logging.getLogger(__name__)

__version__ = "1.0.0"

NAME = "wbdixon"

MODULES = [
    # Maps
    maps.FatFractionDixon(dixon_dir="", ff_name="ff", ff_calc_name="ff_calc"),
    maps.T2starDixon(dixon_dir=""),
    # Segmentations
    segmentations.LegDixon(dixon_dir=""),
    segmentations.TotalSeg(src_dir="", img_glob="water.nii.gz"),
    # Seg fixes
    seg_postprocess.SegFix(
        seg_dir="seg_leg_dixon",
        fix_dir_option="seg_leg_dixon_fix",
        segs={
            "calf_muscle_l.nii.gz": "%s/calf_muscle_l_fixed.nii.gz",
            "calf_muscle_r.nii.gz": "%s/calf_muscle_r_fixed.nii.gz",
            "thigh_muscle_l.nii.gz": "%s/thigh_muscle_l_fixed.nii.gz",
            "thigh_muscle_r.nii.gz": "%s/thigh_muscle_r_fixed.nii.gz",
            "calf_sat_l.nii.gz": "%s/calf_sat_l_fixed.nii.gz",
            "calf_sat_r.nii.gz": "%s/calf_sat_r_fixed.nii.gz",
            "thigh_sat_l.nii.gz": "%s/thigh_sat_l_fixed.nii.gz",
            "thigh_sat_r.nii.gz": "%s/thigh_sat_r_fixed.nii.gz",
        },
        map_dir="seg_leg_dixon",
        map_fname="water.nii.gz",
    ),
    # Statistics
    statistics.SegStats(
        name="stats",
        segs={
            "calf_muscle_r": {
                "dir": "seg_leg_dixon_fix",
                "glob": "calf_muscle_r.nii.gz",
            },
            "calf_muscle_l": {
                "dir": "seg_leg_dixon_fix",
                "glob": "calf_muscle_l.nii.gz",
            },
            "thigh_muscle_r": {
                "dir": "seg_leg_dixon_fix",
                "glob": "thigh_muscle_r.nii.gz",
            },
            "thigh_muscle_l": {
                "dir": "seg_leg_dixon_fix",
                "glob": "thigh_muscle_l.nii.gz",
            },
            "calf_sat_r": {"dir": "seg_leg_dixon_fix", "glob": "calf_sat_r.nii.gz"},
            "calf_sat_l": {"dir": "seg_leg_dixon_fix", "glob": "calf_sat_l.nii.gz"},
            "thigh_sat_r": {"dir": "seg_leg_dixon_fix", "glob": "thigh_sat_r.nii.gz"},
            "thigh_sat_l": {"dir": "seg_leg_dixon_fix", "glob": "thigh_sat_l.nii.gz"},
            "calf_muscle": {"dir": "seg_leg_dixon_fix", "glob": "calf_muscle*.nii.gz"},
            "calf_sat": {"dir": "seg_leg_dixon_fix", "glob": "calf_sat*.nii.gz"},
            "thigh_muscle": {
                "dir": "seg_leg_dixon_fix",
                "glob": "thigh_muscle*.nii.gz",
            },
            "thigh_sat": {"dir": "seg_leg_dixon_fix", "glob": "thigh_sat*.nii.gz"},
            "muscle_r": {"dir": "seg_leg_dixon_fix", "glob": "*muscle_r.nii.gz"},
            "sat_r": {"dir": "seg_leg_dixon_fix", "glob": "*sat_r.nii.gz"},
            "muscle_l": {"dir": "seg_leg_dixon_fix", "glob": "*muscle_l.nii.gz"},
            "sat_l": {"dir": "seg_leg_dixon_fix", "glob": "*sat_l.nii.gz"},
            "total": {"dir": "seg_leg_dixon_fix", "glob": "*.nii.gz"},
            "pancreas": {
                "dir": "totalseg",
                "glob": "pancreas.nii.gz",
                "params": ["ff_calc"],
            },
            "liver": {"dir": "totalseg", "glob": "liver.nii.gz", "params": ["ff_calc"]},
        },
        params={
            "ff_scanner": {
                "dir": "fat_fraction",
                "glob": "ff_scanner.nii.gz",
                "limits": (0, 100),
            },
            "ff_calc": {
                "dir": "fat_fraction",
                "glob": "ff_calc.nii.gz",
                "limits": (0, 100),
            },
            "t2star": {
                "dir": "t2star_dixon",
                "glob": "t2star_exclude_fill.nii.gz",
                "limits": (2, 100),
            },
        },
        stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
        seg_volumes=True,
    ),
]


def add_options(parser):
    parser.add_argument(
        "--seg-leg-dixon-fix", help="Directory containing fixed leg masks"
    )
