# spreadsheets
#  - tkv radiomics, cyst stats definitive, wkv radioms/shapes, pelvis fat volumes, pelvis ff within fat volume, ff within kidney parenchyma
#  - QC cyst stats by different methods, definitive paren + orig paren (model 1 + model 2), pelvis comparison

# DONE Check vat_cor_totalseg should be for everybody?
# DONE? need to add exponential imagetype for t2w radiomics
# DONE? misidentified year 2s
# FIXME additional year 2 oxford subjects (4 - OXF22,OXF10,OXF27, OXF30)
# FIXME CBG05,DER151,DER158,DER50,DER147,NOT048 separate sheet (died, not in afirm baseline but useful for demistifi)
# FIXME _NORM subjects also separate sheet - SF to send list

import logging
import os

import numpy as np

from fsort.image_file import ImageFile
from fproc.module import Module
from fproc.modules import (
    maps,
    segmentations,
    statistics,
    seg_postprocess,
    regrid,
    align,
    misc,
)

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

# Configuration
NAME = "afirm"
STUDYDIR = os.path.join("/gpfs01/spmstore/project/RenalMRI", NAME)
OUTNAME = NAME

COHORTS = [
    ("full cohort", "", None),
    ("baseline cohort", "baseline_", os.path.join(STUDYDIR, "baseline.txt")),
    ("y2 cohort", "y2_", os.path.join(STUDYDIR, "y2_cohort_20260422.txt")),
]

OUTFILES = {
    "kidney": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/stats/stats.csv",
        "fproc/t1_molli_md/t1_molli_md.csv",
    ],
    "t1_radiomics_left": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/t1_radiomics_left/radiomics.csv",
    ],
    "t1_radiomics_right": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/t1_radiomics_right/radiomics.csv",
    ],
    "organs": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/stats_dixon_totalseg/stats.csv",
        "fproc/kidney_dixon_shape_metrics_totalseg_ax/shape_metrics.csv",
        "fproc/kidney_dixon_shape_metrics_totalseg_cor/shape_metrics.csv",
        "fproc/kidney_dixon_radiomics_totalseg_ax/radiomics.csv",
        "fproc/kidney_dixon_radiomics_totalseg_cor/radiomics.csv",
        "fproc/spleen_shape_metrics_totalseg_cor/shape_metrics.csv",
        "fproc/liver_shape_metrics_totalseg_cor/shape_metrics.csv",
        "fproc/pancreas_shape_metrics_totalseg_cor/shape_metrics.csv",
        "fproc/organ_dixon_radiomics/radiomics.csv",
    ],
    "organs_local": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/stats_dixon/stats.csv",
        "fproc/kidney_dixon_shape_metrics_ax/shape_metrics.csv",
        "fproc/kidney_dixon_shape_metrics_cor/shape_metrics.csv",
        "fproc/kidney_dixon_radiomics_ax/radiomics.csv",
        "fproc/kidney_dixon_radiomics_cor/radiomics.csv",
    ],
    "t2w_radiomics": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/t2w_radiomics/radiomics.csv",
    ],
    "shape": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/seg_kidney_t2w_vols/volumes.csv",
        "fproc/tkv_shape_metrics/shape_metrics.csv",
        "fproc/tkv_radiomics/radiomics.csv",
        "fproc/wkv_volumes/stats.csv",
        "fproc/wkv_shape_metrics/shape_metrics.csv",
        "fproc/wkv_radiomics/radiomics.csv",
        "fproc/wkv_ero_volumes/stats.csv",
        "fproc/wkv_ero_shape_metrics/shape_metrics.csv",
        "fproc/wkv_ero_radiomics/radiomics.csv",
        "fproc/kidney_cyst_stats_trace_fixed/kidney_cyst.csv",
        "fproc/kidney_pelvis_stats_trace/stats.csv",
    ],
    "shape_qa": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/kidney_stats_alternate/stats.csv",
        "fproc/kidney_cyst_stats_trace_fixed/kidney_cyst.csv",
        "fproc/kidney_cyst_stats_trace_orig/kidney_cyst.csv",
        "fproc/kidney_cyst_stats/kidney_cyst.csv",
        "fproc/cyst_volume_diffs/diffs.csv",
    ],
    "organ_vols": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/organ_volumes/stats.csv",
    ],
    "totalseg_cor": [
        "fproc/totalseg_cor/volumes.csv",
        "fproc/totalseg_cor/volumes_dilated.csv",
    ],
    "totalseg_ax": [
        "fproc/totalseg_ax/volumes.csv",
        "fproc/totalseg_ax/volumes_dilated.csv",
    ],
}


# FIXME add fat fraction from axial + coronal dixon
class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats",
            default_limits="3t",
            segs={
                "kidney_cortex": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*cortex*.nii.gz",
                },
                "kidney_cortex_l": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*cortex_l*.nii.gz",
                },
                "kidney_cortex_r": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*cortex_r*.nii.gz",
                },
                "kidney_medulla": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*medulla*.nii.gz",
                },
                "kidney_medulla_l": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*medulla_l*.nii.gz",
                },
                "kidney_medulla_r": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*medulla_r*.nii.gz",
                },
                "kidney_l": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*_l.nii.gz",
                    "params": ["b1_stim"],
                },
                "kidney_r": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*_r.nii.gz",
                    "params": ["b1_stim"],
                },
                "tkv_l": {
                    "dir": "seg_kidney_t2w_fix",
                    "glob": "*left*.nii.gz",
                },
                "tkv_r": {
                    "dir": "seg_kidney_t2w_fix",
                    "glob": "*right*.nii.gz",
                },
            },
            params={
                "t2_exp": {
                    "dir": "t2",
                    "glob": "t2_exp.nii.gz",
                    "segs": [
                        "kidney_cortex_l",
                        "kidney_cortex_r",
                        "kidney_medulla_l",
                        "kidney_medulla_r",
                    ],
                },
                "t2_stim": {
                    "dir": "t2",
                    "glob": "t2_stim.nii.gz",
                    "segs": [
                        "kidney_cortex_l",
                        "kidney_cortex_r",
                        "kidney_medulla_l",
                        "kidney_medulla_r",
                    ],
                },
                "b1_stim": {
                    "dir": "t2",
                    "glob": "b1_stim.nii.gz",
                    "segs": [
                        "kidney_cortex_l",
                        "kidney_cortex_r",
                        "kidney_medulla_l",
                        "kidney_medulla_r",
                        "kidney_l",
                        "kidney_r",
                    ],
                },
                "t2star_exp": {
                    "dir": "t2star",
                    "glob": "t2star_2p_exp*.nii.gz",
                },
                "t2star_loglin": {
                    "dir": "t2star",
                    "glob": "t2star_loglin*.nii.gz",
                },
                "r2star_exp": {
                    "dir": "t2star",
                    "glob": "r2star_2p_exp*.nii.gz",
                },
                "r2star_loglin": {
                    "dir": "t2star",
                    "glob": "r2star_loglin*.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_stitch_fix",
                    "glob": "t1_conf.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_clean_native"},
                    },
                },
                "t1_molli_nomdr": {
                    "dir": "t1_molli_nomdr_stitch",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_clean_native"},
                    },
                },
                "t1_molli_mdr": {
                    "dir": "t1_molli_mdr_stitch",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_mdr_clean_native"},
                    },
                },
                "t1_se_nomdr": {
                    "dir": "t1_se_nomdr_stitch",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_se_clean_native"},
                    },
                },
                "t1_se_mdr_2p": {
                    "dir": "t1_se_mdr_stitch",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_se_clean_native"},
                    },
                },
                "t1_se_mdr_3p": {
                    "dir": "t1_se_mdr_step2_stitch",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_se_clean_native"},
                    },
                },
                "mtr": {
                    "dir": "mtr",
                    "glob": "mtr.nii.gz",
                },
                "mtr_mdr": {
                    "dir": "mtr",
                    "glob": "mtr.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_mdr_clean_native"},
                    },
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                    "segs": ["tkv_l", "tkv_r"],
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                    "segs": ["tkv_l", "tkv_r"],
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                    "segs": ["tkv_l", "tkv_r"],
                },
                "adc_mdr": {
                    "dir": "dwi_adc",
                    "glob": "*adc_map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_se_clean_native"},
                    },
                },
                "ff_cor": {
                    "dir": "ff_dixon_cor",
                    "glob": "fat_fraction.nii.gz",
                },
                "ff_ax": {
                    "dir": "ff_dixon_ax",
                    "glob": "fat_fraction.nii.gz",
                },
            },
            stats=[
                "n",
                "vol",
                "iqn",
                "iqvol",
                "iqmean",
                "median",
                "iqstd",
                "perc90",
                "te",
                "mode",
                "fwhm",
            ],
        )


class StatsDixon(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats_dixon",
            default_limits="3t",
            segs={
                "liver_cor": {
                    "dir": "seg_liver_dixon_cor",
                    "glob": "liver.nii.gz",
                    "seg_volumes": False,
                },
                "spleen_cor": {
                    "dir": "seg_spleen_dixon_cor",
                    "glob": "spleen.nii.gz",
                    "seg_volumes": False,
                },
                "sat_cor": {
                    "dir": "seg_sat_dixon_cor",
                    "glob": "sat.nii.gz",
                    "seg_volumes": False,
                },
                "liver_ax": {"dir": "seg_liver_dixon_ax", "glob": "liver.nii.gz"},
                "spleen_ax": {"dir": "seg_spleen_dixon_ax", "glob": "spleen.nii.gz"},
                "sat_ax": {
                    "dir": "seg_sat_dixon_ax",
                    "glob": "sat.nii.gz",
                },
                "vat_ax": {
                    "dir": "seg_vat_dixon_ax_local",
                    "glob": "vat.nii.gz",
                    "params": [],  # Volumes only
                },
                "pancreas": {
                    "dir": "seg_pancreas_ethrive",
                    "glob": "pancreas.nii.gz",
                },
                "kidney_dixon_nofat_cor": {
                    "dir": "seg_kidney_fat_dixon_cor",
                    "glob": "kidney_parenchyma.nii.gz",
                    "params": [],
                },
                "kidney_dixon_left_nofat_cor": {
                    "dir": "seg_kidney_fat_dixon_cor",
                    "glob": "kidney_parenchyma_left.nii.gz",
                    "params": [],
                },
                "kidney_dixon_right_nofat_cor": {
                    "dir": "seg_kidney_fat_dixon_cor",
                    "glob": "kidney_parenchyma_right.nii.gz",
                    "params": [],
                },
                "kidney_dixon_nofat_ax": {
                    "dir": "seg_kidney_fat_dixon_ax",
                    "glob": "kidney_parenchyma.nii.gz",
                    "params": [],
                },
                "kidney_dixon_left_nofat_ax": {
                    "dir": "seg_kidney_fat_dixon_ax",
                    "glob": "kidney_parenchyma_left.nii.gz",
                    "params": [],
                },
                "kidney_dixon_right_nofat_ax": {
                    "dir": "seg_kidney_fat_dixon_ax",
                    "glob": "kidney_parenchyma_right.nii.gz",
                    "params": [],
                },
                "fat_pelvis_cor": {
                    "dir": "seg_kidney_fat_dixon_cor",
                    "glob": "fat_pelvis.nii.gz",
                    "params": ["ff_cor"],
                },
                "fat_pelvis_left_cor": {
                    "dir": "seg_kidney_fat_dixon_cor",
                    "glob": "fat_pelvis_left.nii.gz",
                    "params": ["ff_cor"],
                },
                "fat_pelvis_right_cor": {
                    "dir": "seg_kidney_fat_dixon_cor",
                    "glob": "fat_pelvis_right.nii.gz",
                    "params": ["ff_cor"],
                },
                "fat_pelvis_ax": {
                    "dir": "seg_kidney_fat_dixon_ax",
                    "glob": "fat_pelvis.nii.gz",
                    "params": ["ff_ax"],
                },
                "fat_pelvis_left_ax": {
                    "dir": "seg_kidney_fat_dixon_ax",
                    "glob": "fat_pelvis_left.nii.gz",
                    "params": ["ff_ax"],
                },
                "fat_pelvis_right_ax": {
                    "dir": "seg_kidney_fat_dixon_ax",
                    "glob": "fat_pelvis_right.nii.gz",
                    "params": ["ff_ax"],
                },
            },
            params={
                "t2star_exp": {
                    "dir": "t2star",
                    "glob": "t2star*_exp.nii.gz",
                },
                "r2star_exp": {
                    "dir": "t2star",
                    "glob": "r2star*_exp.nii.gz",
                },
                "t2star_loglin": {
                    "dir": "t2star",
                    "glob": "t2star*_loglin.nii.gz",
                },
                "r2star_loglin": {
                    "dir": "t2star",
                    "glob": "r2star*_loglin.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_stitch",
                    "glob": "t1_conf.nii.gz",
                },
                "t1_ax": {
                    "dir": "molli_ax",
                    "src": self.INPUT,
                    "glob": "t1_conf.nii.gz",
                },
                "mtr": {
                    "dir": "mtr",
                    "glob": "mtr.nii.gz",
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                },
                "t2_exp": {
                    "dir": "t2",
                    "glob": "t2_exp.nii.gz",
                },
                "t2_stim": {
                    "dir": "t2",
                    "glob": "t2_stim.nii.gz",
                },
                "b1_stim": {
                    "dir": "t2",
                    "glob": "b1_stim.nii.gz",
                },
                "ff_cor": {
                    "dir": "ff_dixon_cor",
                    "glob": "fat_fraction.nii.gz",
                },
                "ff_ax": {
                    "dir": "ff_dixon_ax",
                    "glob": "fat_fraction.nii.gz",
                },
            },
            stats=[
                "n",
                "vol",
                "iqn",
                "iqvol",
                "iqmean",
                "median",
                "iqstd",
                "perc90",
                "te",
                "mode",
                "fwhm",
            ],
            seg_volumes=True,
        )


class StatsDixonTotalseg(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats_dixon_totalseg",
            default_limits="3t",
            segs={
                "liver_cor": {
                    "dir": "totalseg_cor",
                    "glob": "liver.nii.gz",
                    "seg_volumes": False,
                },
                "spleen_cor": {
                    "dir": "totalseg_cor",
                    "glob": "spleen.nii.gz",
                    "seg_volumes": False,
                },
                "sat_cor": {
                    "dir": "totalseg_cor",
                    "glob": "subcutaneous_fat.nii.gz",
                    "seg_volumes": False,
                },
                "liver_ax": {"dir": "totalseg_ax", "glob": "liver.nii.gz"},
                "spleen_ax": {"dir": "totalseg_ax", "glob": "spleen.nii.gz"},
                "sat_ax": {
                    "dir": "totalseg_ax",
                    "glob": "subcutaneous_fat.nii.gz",
                },
                "vat_ax": {
                    "dir": "seg_vat_dixon_ax_totalseg",
                    "glob": "vat.nii.gz",
                    "params": [],  # Volumes only
                },
                "pancreas": {
                    "dir": "totalseg_cor",
                    "glob": "pancreas.nii.gz",
                },
                "kidney_dixon_nofat_cor": {
                    "dir": "seg_kidney_fat_dixon_totalseg_cor",
                    "glob": "kidney_parenchyma.nii.gz",
                    "params": [],
                },
                "kidney_dixon_left_nofat_cor": {
                    "dir": "seg_kidney_fat_dixon_totalseg_cor",
                    "glob": "kidney_parenchyma_left.nii.gz",
                    "params": [],
                },
                "kidney_dixon_right_nofat_cor": {
                    "dir": "seg_kidney_fat_dixon_totalseg_cor",
                    "glob": "kidney_parenchyma_right.nii.gz",
                    "params": [],
                },
                "kidney_dixon_nofat_ax": {
                    "dir": "seg_kidney_fat_dixon_totalseg_ax",
                    "glob": "kidney_parenchyma.nii.gz",
                    "params": [],
                },
                "kidney_dixon_left_nofat_ax": {
                    "dir": "seg_kidney_fat_dixon_totalseg_ax",
                    "glob": "kidney_parenchyma_left.nii.gz",
                    "params": [],
                },
                "kidney_dixon_right_nofat_ax": {
                    "dir": "seg_kidney_fat_dixon_totalseg_ax",
                    "glob": "kidney_parenchyma_right.nii.gz",
                    "params": [],
                },
                "fat_pelvis_cor": {
                    "dir": "seg_kidney_fat_dixon_totalseg_cor",
                    "glob": "fat_pelvis.nii.gz",
                    "params": ["ff_cor"],
                },
                "fat_pelvis_left_cor": {
                    "dir": "seg_kidney_fat_dixon_totalseg_cor",
                    "glob": "fat_pelvis_left.nii.gz",
                    "params": ["ff_cor"],
                },
                "fat_pelvis_right_cor": {
                    "dir": "seg_kidney_fat_dixon_totalseg_cor",
                    "glob": "fat_pelvis_right.nii.gz",
                    "params": ["ff_cor"],
                },
                "fat_pelvis_ax": {
                    "dir": "seg_kidney_fat_dixon_totalseg_ax",
                    "glob": "fat_pelvis.nii.gz",
                    "params": ["ff_ax"],
                },
                "fat_pelvis_left_ax": {
                    "dir": "seg_kidney_fat_dixon_totalseg_ax",
                    "glob": "fat_pelvis_left.nii.gz",
                    "params": ["ff_ax"],
                },
                "fat_pelvis_right_ax": {
                    "dir": "seg_kidney_fat_dixon_totalseg_ax",
                    "glob": "fat_pelvis_right.nii.gz",
                    "params": ["ff_ax"],
                },
            },
            params={
                "t2star_exp": {
                    "dir": "t2star",
                    "glob": "t2star*_exp.nii.gz",
                },
                "r2star_exp": {
                    "dir": "t2star",
                    "glob": "r2star*_exp.nii.gz",
                },
                "t2star_loglin": {
                    "dir": "t2star",
                    "glob": "t2star*_loglin.nii.gz",
                },
                "r2star_loglin": {
                    "dir": "t2star",
                    "glob": "r2star*_loglin.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_stitch",
                    "glob": "t1_conf.nii.gz",
                },
                "t1_ax": {
                    "dir": "molli_ax",
                    "src": self.INPUT,
                    "glob": "t1_conf.nii.gz",
                },
                "mtr": {
                    "dir": "mtr",
                    "glob": "mtr.nii.gz",
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                },
                "t2_exp": {
                    "dir": "t2",
                    "glob": "t2_exp.nii.gz",
                },
                "t2_stim": {
                    "dir": "t2",
                    "glob": "t2_stim.nii.gz",
                },
                "b1_stim": {
                    "dir": "t2",
                    "glob": "b1_stim.nii.gz",
                },
                "ff_cor": {
                    "dir": "ff_dixon_cor",
                    "glob": "fat_fraction.nii.gz",
                },
                "ff_ax": {
                    "dir": "ff_dixon_ax",
                    "glob": "fat_fraction.nii.gz",
                },
            },
            stats=[
                "n",
                "vol",
                "iqn",
                "iqvol",
                "iqmean",
                "median",
                "iqstd",
                "perc90",
                "te",
                "mode",
                "fwhm",
            ],
            seg_volumes=True,
        )


class T1MolliMetadata(Module):
    def __init__(self, name="t1_molli_md", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t1_dir = "t1_molli"
        t1_glob = "t1_*.nii.gz"
        t1s = self.inimgs(t1_dir, t1_glob, src=self.INPUT)
        if not t1s:
            self.no_data(f"No T1 maps found in {t1_dir} matching {t1_glob}")

        tis = []
        hr = []
        for t1 in t1s:
            tis.extend(list(t1.inversiontimedelay))
            hr.extend(list(t1.heartrate))

        hr = np.unique(hr)
        if len(hr) > 1:
            LOG.warn(f"Multiple heart rates found: {hr} - using first")
            hr = hr[0]
        elif len(hr) == 0:
            LOG.warn("No heart rate found")
            hr = ""
        else:
            hr = hr[0]
            LOG.info(f" - Found heart rate: {hr}")

        tis = sorted([float(v) for v in np.unique(tis) if float(v) > 0])
        with open(self.outfile("tis.txt"), "w") as f:
            f.write("\n".join([str(v) for v in tis]))

        LOG.info(f" - Found TIs: {tis}")
        if len(tis) >= 3:
            ti1, ti2, spacing = tis[0], tis[1], tis[2] - tis[0]
        else:
            ti1, ti2, spacing = "", "", ""
            LOG.warn(f"Not enough TIs found: {tis}")

        with open(self.outfile("t1_molli_md.csv"), "w") as f:
            f.write(f"t1_molli_heart_rate,{hr}\n")
            f.write(f"t1_molli_ti1,{ti1}\n")
            f.write(f"t1_molli_ti2,{ti2}\n")
            f.write(f"t1_molli_ti_spacing,{spacing}\n")


class T1Scaled(Module):

    def __init__(self, name="t1_scaled", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        scale_factors_fname = self.pipeline.options.t1_scale_factors
        sf_left, sf_right = None, None
        if scale_factors_fname and os.path.isfile(scale_factors_fname):
            with open(scale_factors_fname) as f:
                for l in f.readlines()[1:]:
                    parts = l.split(",")
                    if len(parts) != 3:
                        LOG.warn(f"Invalid line in scale factors file: {l}")
                        continue
                    subjid = parts[0]
                    if (
                        subjid.strip().lower()
                        == self.pipeline.options.subjid.strip().lower()
                    ):
                        sf_left, sf_right = float(parts[1]), float(parts[2])
                        break

        if not sf_left:
            LOG.warn(
                f"Scale factor (left) not found for {self.pipeline.options.subjid} - will output unscaled data"
            )
            sf_left = 1
        if not sf_right:
            LOG.warn(
                f"Scale factor (right) not found for {self.pipeline.options.subjid} - will output unscaled data"
            )
            sf_right = 1

        t1_dir = "t1_molli_stitch_fix"
        t1_glob = "t1_*.nii.gz"
        t1s = self.inimgs(t1_dir, t1_glob, src=self.OUTPUT)
        if not t1s:
            self.no_data(f"No T1 maps found in {t1_dir} matching {t1_glob}")

        LOG.info(" - T1 scale factors (left/right) : %.4f / %.4f" % (sf_left, sf_right))
        for t1 in t1s:
            left_data = t1.data * sf_left
            right_data = t1.data * sf_right
            t1.save_derived(
                left_data,
                self.outfile(t1.fname.replace(".nii.gz", "_scaled_left.nii.gz")),
            )
            t1.save_derived(
                right_data,
                self.outfile(t1.fname.replace(".nii.gz", "_scaled_right.nii.gz")),
            )


class SegKidneyCystTraceData(Module):

    def __init__(self, name="seg_kidney_cyst_t2w_trace", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        srcdir = self.pipeline.options.seg_kidney_cyst_trace
        orig_fpath = f"{srcdir}/{self.pipeline.options.subjid}_ORIG_mask.nii.gz"
        if not os.path.exists(orig_fpath):
            LOG.warn(
                f"No original TRACE cyst seg found for {self.pipeline.options.subjid} in {orig_fpath}"
            )
            orig = None
        else:
            orig = ImageFile(orig_fpath, warn_json=False)
            LOG.info(" - Saving original TRACE cyst seg to kidney_cyst_orig.nii.gz")
            orig.save(self.outfile("kidney_cyst_orig.nii.gz"))

        fixed_fpath = f"{srcdir}/{self.pipeline.options.subjid}_FIX_mask.nii.gz"
        if not os.path.exists(fixed_fpath):
            if orig is not None:
                LOG.warn(
                    f" - No fixed TRACE cyst seg found for {self.pipeline.options.subjid}, copying original"
                )
                orig.save(self.outfile("kidney_cyst_fixed.nii.gz"))
            else:
                LOG.warn(
                    f"No fixed TRACE cyst seg found for {self.pipeline.options.subjid} in {fixed_fpath}"
                )
        else:
            fixed = ImageFile(fixed_fpath, warn_json=False)
            LOG.info(" - Saving fixed TRACE cyst seg to kidney_cyst_fixed.nii.gz")
            fixed.save(self.outfile("kidney_cyst_fixed.nii.gz"))


class SegOrgansTraceData(Module):

    def __init__(self, name="seg_organs_trace", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        srcdir = self.pipeline.options.seg_organs_trace
        fpath = f"{srcdir}/{self.pipeline.options.subjid}.nii.gz"
        if not os.path.exists(fpath):
            self.no_data(
                f"No TRACE cyst organ seg found for {self.pipeline.options.subjid} in {fpath}"
            )

        img = ImageFile(fpath, warn_json=False)
        LOG.info(" - Saving TRACE organ seg to trace_organs.nii.gz")
        img.save(self.outfile("trace_organs.nii.gz"))

        with open(self.outfile("volumes.csv"), "w") as f:
            for idx, name in {
                1: "trace_kidney_r",
                2: "trace_kidney_l",
                3: "trace_spleen",
                4: "trace_liver",
            }.items():
                roi = (img.data == idx).astype(np.int8)
                organ_img = img.save_derived(roi, self.outfile(f"{name}.nii.gz"))
                self.lightbox(img, organ_img, name=f"{name}_overlay", tight=True)
                volume = np.count_nonzero(roi) * img.voxel_volume
                LOG.info(f" - {name}: volume = {volume} mL")
                f.write(f"{name},{volume}\n")
            tkv = np.logical_or(img.data == 1, img.data == 2)
            tkv_img = img.save_derived(
                tkv.astype(np.int8), self.outfile("trace_kidney_all.nii.gz")
            )
            self.lightbox(img, tkv_img, name="trace_kidney_all_overlay", tight=True)
            tkv_volume = np.count_nonzero(tkv) * img.voxel_volume
            LOG.info(f" - Total kidney volume: {tkv_volume} mL")
            f.write(f"trace_kidney_all,{tkv_volume}\n")


class SegKidneyWhole(Module):

    def __init__(self, name="seg_kidney_whole", **kwargs):
        Module.__init__(self, name, deps=["seg_organs_trace", "traceseg"], **kwargs)

    def process(self):
        trace_seg = self.single_inimg(
            "seg_organs_trace", "trace_kidney_all.nii.gz", src=self.OUTPUT
        )
        if not trace_seg:
            self.no_data(
                "No TRACE (new) kidney seg found in seg_organs_trace/trace_kidney_all.nii.gz"
            )

        SUBJIDS_USE_OLD_TRACE_SEG = [
            "CBG_023_V1",
            "CBG_028_V1",
            "DER_006_V1",
            "DER_062_V1",
            "DER_135_V1",
            "EDB_109_V1",
            "EDB_116_V1",
            "EDB_144_V1",
            "GLA_002_V1",
            "LDS_045_V1",
            "LDS_114_V1",
            "MAN_008_V1",
            "MAN_014_V1",
            "OXF_021_V1",
        ]
        if self.pipeline.options.subjid in SUBJIDS_USE_OLD_TRACE_SEG:
            trace_seg = self.single_inimg(
                "traceseg", "trace_kidney_all.nii.gz", src=self.OUTPUT
            )
            if not trace_seg:
                self.no_data(
                    f"No TRACE (old) kidney seg found for {self.pipeline.options.subjid}"
                )

        trace_seg.reorient2std()
        # tkv = self.single_inimg("seg_kidney_t2w_fix", "kidney_mask.nii.gz", src=self.OUTPUT)
        # if not tkv:
        #    LOG.warning("No TKV kidney seg found in seg_kidney_t2w - will not add to whole kidney")
        #    tkv_data = np.zeros_like(trace_new.data)
        # else:
        #    tkv.reorient2std()
        #    tkv_data = tkv.data
        # combined_data = np.logical_or(trace_new.data > 0, tkv_data > 0).astype(np.int8)
        # combined_img = trace_new.save_derived(combined_data, self.outfile("kidney_whole.nii.gz"))

        trace_seg.save(self.outfile("kidney_whole.nii.gz"))
        left_data = self.split_lr(trace_seg.data, trace_seg.affine, side="l")
        trace_seg.save_derived(left_data, self.outfile("kidney_whole_l.nii.gz"))
        right_data = self.split_lr(trace_seg.data, trace_seg.affine, side="r")
        trace_seg.save_derived(right_data, self.outfile("kidney_whole_r.nii.gz"))

        t2w = self.single_inimg("../fsort/t2w", "t2w.nii.gz")
        if t2w:
            t2w.reorient2std()
            self.lightbox(t2w, trace_seg, name="kidney_whole_overlay", tight=True)


class TempAddPelvis(Module):
    """
    Temp module to add manual pelvis segmentations from TRACE data into organ seg
    """

    def __init__(self, name="temp_add_pelvis", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        subjid = self.pipeline.options.subjid
        pelvis_fpath = os.path.join(
            f"/spmstore/project/RenalMRI/afirm/PELVIS_RENAMED/{subjid}.nii.gz"
        )
        organ_fpath = os.path.join(
            f"/spmstore/project/RenalMRI/afirm/TRACE_ORGANS_RENAMED/{subjid}.nii.gz"
        )
        if not os.path.exists(pelvis_fpath) or not os.path.exists(organ_fpath):
            self.no_data(
                f"No pelvis seg found for {self.pipeline.options.subjid} in {pelvis_fpath} or {organ_fpath}"
            )

        pelvis_img = ImageFile(pelvis_fpath, warn_json=False)
        pelvis_img.reorient2std()
        from scipy.ndimage import binary_dilation

        pelvis_data = binary_dilation((pelvis_img.data > 0).astype(np.int8))
        pevis_img_l = self.split_lr(pelvis_data, pelvis_img.affine, side="l")
        pevis_img_r = self.split_lr(pelvis_data, pelvis_img.affine, side="r")
        organ_img = ImageFile(organ_fpath, warn_json=False)
        organ_img.reorient2std()
        organ_data = np.copy(organ_img.data.astype(np.int32))
        organ_data[pevis_img_l > 0] = 2
        organ_data[pevis_img_r > 0] = 1
        organ_img.save_derived(organ_data, self.outfile(f"{subjid}.nii.gz"))


class DixonCorBest(Module):
    """
    Select best coronal dixon images
    """

    def __init__(self, name="dixon_cor_best", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        mdixon_imgs = self.inimgs("../fsort/dixon_cor", "*.nii.gz", src=self.INPUT)
        if mdixon_imgs:
            LOG.info(" - Found coronal mdixon images")
            for img in mdixon_imgs:
                img.save(self.outfile(img.fname))
        else:
            LOG.info(
                " - No coronal mdixon images found, looking for generic dixon images"
            )
            dixon_imgs = self.inimgs(
                "../fsort/dixon_generic", "*.nii.gz", src=self.INPUT
            )
            if not dixon_imgs:
                self.no_data("No generic dixon images found")
            for img in dixon_imgs:
                img.save(self.outfile(img.fname))
            if self.pipeline.options.subjid in ["LDS_026_V1"]:
                LOG.info(" - Subject with known fat/water swap - changing filenames")
                water_img = ImageFile(self.outfile("fat.nii.gz"))
                fat_img = ImageFile(self.outfile("water.nii.gz"))
                water_data, fat_data = water_img.data, fat_img.data
                water_img.save_derived(water_data, self.outfile("water.nii.gz"))
                fat_img.save_derived(fat_data, self.outfile("fat.nii.gz"))


MODULES = [
    misc.ScanDates(
        "scan_dates",
        input={
            "../fsort/t1w": "*.nii.gz",
            "../fsort/t2w": "*.nii.gz",
            "../fsort/t1_molli": "*.nii.gz",
        },
    ),
    maps.DixonClassify(
        dixon_src="../fsort/raw_dixon",
        fixes="/spmstore/project/RenalMRI/afirm/dixon_classify_fixes.csv",
    ),
    DixonCorBest(),
    # Parameter maps
    maps.T1Molli(
        name="t1_molli",
        molli_dir="../fsort/t1_molli",
        molli_glob="t1_molli_raw*.nii.gz",
        t1_thresh=(0, 5000),
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1Molli(
        name="t1_molli_mdr",
        molli_dir="../fsort/t1_molli_raw",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=True,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1Molli(
        name="t1_molli_nomdr",
        molli_dir="../fsort/t1_molli_raw",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=False,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1SE(
        name="t1_se_nomdr",
        se_dir="../fsort/t1_se_raw",
        tis=np.arange(100, 2001, 100),
        tss=53.7,
        mag_only=True,
    ),
    maps.T1SE(
        name="t1_se_mdr",
        se_dir="t1_se_raw",
        tis=np.arange(100, 2001, 100),
        tss=53.7,
        mdr=True,
        mag_only=True,
        parameters=2,
    ),
    maps.T1SE(
        name="t1_se_mdr_step2",
        deps=["t1_se_mdr"],
        se_dir="t1_se_mdr",
        tis=np.arange(100, 2001, 100),
        tss=53.7,
        se_mag_glob="*_reg.nii.gz",
        mdr=True,
        mag_only=True,
        parameters=3,
        se_src=Module.OUTPUT,
    ),
    maps.T2(),
    maps.T2star(),
    maps.MTR(),
    maps.B0(),
    maps.B1(),
    maps.FatFractionDixon(name="ff_dixon_cor", dixon_dir="../fproc/dixon_cor_best"),
    maps.FatFractionDixon(name="ff_dixon_ax", dixon_dir="dixon_ax"),
    maps.DwiMoco(),
    maps.DwiAdc(),
    maps.AslMoco(name="pcasl_moco", asl_glob="pcasl*.nii.gz"),
    maps.AslMoco(name="fair_moco", asl_glob="fair*.nii.gz"),
    # Stitch together potentially multiple slice maps
    regrid.StitchSlices(
        name="t1_molli_stitch",
        img_dir="t1_molli",
        imgs={
            "*t1_map*.nii.g?": "t1_map.nii.gz",  # Just to make sure globs are unique keys!
            "*t1_map*.nii.gz": "t1_conf.nii.gz",
            "*t1_conf*.nii.gz": "t1_conf.nii.gz",
        },
    ),
    regrid.StitchSlices(
        name="t1_molli_nomdr_stitch",
        img_dir="t1_molli_nomdr",
        imgs={
            "*t1_map*.nii.gz": "t1_map.nii.gz",
        },
    ),
    regrid.StitchSlices(
        name="t1_molli_mdr_stitch",
        img_dir="t1_molli_mdr",
        imgs={
            "*t1_map*.nii.gz": "t1_map.nii.gz",
        },
    ),
    regrid.StitchSlices(
        name="t1_se_nomdr_stitch",
        img_dir="t1_se_nomdr",
        imgs={
            "*t1_map*.nii.gz": "t1_map.nii.gz",
        },
    ),
    regrid.StitchSlices(
        name="t1_se_mdr_stitch",
        img_dir="t1_se_mdr",
        imgs={
            "*t1_map*.nii.gz": "t1_map.nii.gz",
        },
    ),
    regrid.StitchSlices(
        name="t1_se_mdr_step2_stitch",
        img_dir="t1_se_mdr_step2",
        imgs={
            "*t1_map*.nii.gz": "t1_map.nii.gz",
            "*_reg_reg*.nii.gz": "se_data.nii.gz",
        },
    ),
    # Segmentations
    segmentations.KidneyT1(
        map_dir="t1_molli_stitch",
        map_glob="t1_conf.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT1(
        name="seg_kidney_t1_mdr",
        map_dir="t1_molli_mdr_stitch",
        map_glob="t1_map.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT1(
        name="seg_kidney_t1_nomdr",
        map_dir="t1_molli_nomdr_stitch",
        map_glob="t1_map.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT1SE(
        name="seg_kidney_t1_se",
        t1_se_dir="t1_se_mdr_step2_stitch",
        t1_se_glob="*se_data.nii.gz",
    ),
    segmentations.KidneyT2wRenalSegmentor(name="seg_kidney_t2w"),
    segmentations.KidneyT2w(name="seg_kidney_t2w_model2"),
    segmentations.KidneyT2w(
        name="seg_kidney_t2w_model1",
        model="/software/imaging/body_pipelines/trained_models/t2w_seg.h5",
    ),
    segmentations.KidneyCystT2w(
        t2w_dir="t2w", t2w_glob="t2w.nii.gz", t2w_src=Module.INPUT
    ),
    SegKidneyCystTraceData(),
    SegOrgansTraceData(),
    # segmentations.BodyDixon(),
    segmentations.SatDixon(
        name="seg_sat_dixon_cor", dixon_dir="../fproc/dixon_cor_best"
    ),
    segmentations.LiverDixon(
        name="seg_liver_dixon_cor", dixon_dir="../fproc/dixon_cor_best"
    ),
    segmentations.SpleenDixon(
        name="seg_spleen_dixon_cor", dixon_dir="../fproc/dixon_cor_best"
    ),
    segmentations.KidneyDixon(
        name="seg_kidney_dixon_cor", dixon_dir="../fproc/dixon_cor_best", model_id="422"
    ),
    segmentations.SatDixon(name="seg_sat_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.LiverDixon(name="seg_liver_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.SpleenDixon(name="seg_spleen_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.KidneyDixon(
        name="seg_kidney_dixon_ax", dixon_dir="dixon_ax", model_id="422"
    ),
    segmentations.PancreasEthrive(),
    segmentations.BodyDixon(name="seg_body_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.BodyDixon(
        name="seg_body_dixon_cor", dixon_dir="../fproc/dixon_cor_best"
    ),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive", "pancreas.nii.gz"),
    segmentations.VatDixon(
        name="seg_vat_dixon_ax_local",
        ff_dir="ff_dixon_ax",
        ff_glob="fat_fraction.nii.gz",
        body_dir="seg_body_dixon_ax",
        sat_dir="seg_sat_dixon_ax",
        organs={
            "seg_liver_dixon_ax": "liver.nii.gz",
            "seg_spleen_dixon_ax": "spleen.nii.gz",
            "seg_pancreas_ethrive_largestblob": "pancreas.nii.gz",
            "seg_kidney_dixon_ax": "kidney.nii.gz",
        },
    ),
    segmentations.VatDixon(
        name="seg_vat_dixon_cor_local",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        body_dir="seg_body_dixon_cor",
        sat_dir="seg_sat_dixon_cor",
        organs={
            "seg_liver_dixon_cor": "liver.nii.gz",
            "seg_spleen_dixon_cor": "spleen.nii.gz",
            "seg_pancreas_ethrive_largestblob": "pancreas.nii.gz",
            "seg_kidney_dixon_cor": "kidney.nii.gz",
        },
    ),
    segmentations.TotalSeg(
        name="totalseg_cor",
        src_dir="../fproc/dixon_cor_best",
        dilate=1,
        csv_suffix="_cor",
    ),
    segmentations.TotalSeg(
        name="totalseg_ax", src_dir="dixon_ax", dilate=1, csv_suffix="_ax"
    ),
    segmentations.TotalSeg(
        name="totalseg_sag_loc",
        src_dir="sag_local",
        water_glob="sag_local.nii.gz",
        fat_glob=None,
        dilate=1,
        csv_suffix="_sagloc",
    ),
    segmentations.VatDixon(
        name="seg_vat_dixon_cor_totalseg",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        body_dir="seg_body_dixon_cor",
        sat_dir="totalseg_cor",
        sat_glob="subcutaneous_fat.nii.gz",
        dixon_dir="../fproc/dixon_cor_best",
        organs={
            "totalseg_cor": "liver.nii.gz",
            "totalseg_cor": "spleen.nii.gz",
            "totalseg_cor": "pancreas.nii.gz",
            "totalseg_cor": "kidneys.nii.gz",
        },
    ),
    segmentations.VatDixon(
        name="seg_vat_dixon_ax_totalseg",
        ff_dir="ff_dixon_ax",
        ff_glob="fat_fraction.nii.gz",
        body_dir="seg_body_dixon_ax",
        sat_dir="totalseg_ax",
        sat_glob="subcutaneous_fat.nii.gz",
        dixon_dir="../fproc/dixon_cor_best",
        organs={
            "totalseg_ax": "liver.nii.gz",
            "totalseg_ax": "spleen.nii.gz",
            "totalseg_ax": "pancreas.nii.gz",
            "totalseg_ax": "kidneys.nii.gz",
        },
    ),
    regrid.CombineSegs(
        name="seg_liver_combined",
        seg_dirs=["../fproc/totalseg_cor", "../fproc/totalseg_sag_loc"],
        seg_globs=["liver.nii.gz", "liver.nii.gz"],
        out_fname="liver.nii.gz",
    ),
    regrid.CombineSegs(
        name="seg_pancreas_combined",
        seg_dirs=["../fproc/totalseg_cor", "../fproc/totalseg_sag_loc"],
        seg_globs=["pancreas.nii.gz", "pancreas.nii.gz"],
        out_fname="pancreas.nii.gz",
    ),
    regrid.CombineSegs(
        name="seg_spleen_combined",
        seg_dirs=["../fproc/totalseg_cor", "../fproc/totalseg_sag_loc"],
        seg_globs=["spleen.nii.gz", "spleen.nii.gz"],
        out_fname="spleen.nii.gz",
    ),
    segmentations.TraceSeg(name="traceseg", src_dir="t2w", img_glob="t2w.nii.gz"),
    SegKidneyWhole(),
    # Manual fixes
    seg_postprocess.SplitLR(
        "seg_kidney_t1_se",
        "*kidney*.nii.gz",
    ),
    maps.MapFix(
        "t1_molli_stitch",
        fix_dir_option="seg_kidney_t1_fix",
        maps={
            "t1_map.nii.gz": {
                "glob": "%s/t1_map.nii.gz",
            },
            "t1_conf.nii.gz": {
                "glob": "%s/t1_map.nii.gz",
            },
        },
    ),
    T1Scaled(),
    seg_postprocess.SegFix(
        "seg_kidney_t1",
        fix_dir_option="seg_kidney_t1_fix",
        segs={
            "*cortex_l*.nii.gz": {
                "glob": "%s/*cortex*.nii.gz",
                "side": "left",
                "fname": "kidney_cortex_l.nii.gz",
            },
            "*cortex_r*.nii.gz": {
                "glob": "%s/*cortex*.nii.gz",
                "side": "right",
                "fname": "kidney_cortex_r.nii.gz",
            },
            "*medulla_l*.nii.gz": {
                "glob": "%s/*medulla*.nii.gz",
                "side": "left",
                "fname": "kidney_medulla_l.nii.gz",
            },
            "*medulla_r*.nii.gz": {
                "glob": "%s/*medulla*.nii.gz",
                "side": "right",
                "fname": "kidney_medulla_r.nii.gz",
            },
        },
        map_dir="t1_molli_stitch_fix",
        map_fname="t1_map.nii.gz",
    ),
    seg_postprocess.SegFix(
        "seg_kidney_cyst_t2w",
        fix_dir_option="cyst_masks",
        segs={
            "kidney_cyst_mask.nii.gz": "%s/*FIX*.nii.gz",
        },
        map_dir="t2w",
        map_fname="t2w.nii.gz",
        map_src=Module.INPUT,
    ),
    seg_postprocess.SegFix(
        "seg_kidney_t2w",
        fix_dir_option="seg_kidney_t2w_fix",
        segs={
            "*mask*.nii.gz": {
                "fname": "kidney_mask.nii.gz",
                "glob": "%s.nii.gz",
            },
            "*left*.nii.gz": {
                "fname": "kidney_left.nii.gz",
                "glob": "%s.nii.gz",
                "side": "left",
            },
            "*right*.nii.gz": {
                "fname": "kidney_right.nii.gz",
                "glob": "%s.nii.gz",
                "side": "right",
            },
        },
        map_dir="t2w",
        map_fname="t2w.nii.gz",
        map_src=Module.INPUT,
    ),
    # Re-alignments
    align.FlirtAlignOnly(
        name="seg_kidney_t1_align_t2star",
        in_dir="t1_molli_stitch_fix",
        in_glob="t1_map.nii.gz",
        ref_dir="t2star",
        ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w_fix",
        weight_mask="kidney_mask.nii.gz",
        weight_mask_dil=6,
        also_align={
            "seg_kidney_t1_fix": "kidney*.nii.gz",
            "t1_molli_stitch_fix": "t1_conf.nii.gz",
        },
    ),
    # Segmentation cleaning
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean",
        srcdir="seg_kidney_t1_align_t2star",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="seg_kidney_t1_align_t2star",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean_native",
        srcdir="seg_kidney_t1_fix",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_stitch_fix",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_mdr_clean_native",
        srcdir="seg_kidney_t1_mdr",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_mdr_stitch",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_se_clean_native",
        srcdir="seg_kidney_t1_se_splitlr",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_se_mdr_step2_stitch",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean_native_generic",
        srcdir="seg_kidney_t1_fix",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_stitch_fix",
        t1_map_glob="t1_map.nii.gz",
        t2w=False,
    ),
    seg_postprocess.KidneyCystClean(
        cyst_dir="seg_kidney_cyst_t2w_fix",
        seg_t2w_dir="seg_kidney_t2w_fix",
        t2w_dir="t2w",
        t2w_glob="t2w.nii.gz",
        t2w_src=Module.INPUT,
    ),
    seg_postprocess.SegVolumes(
        "seg_kidney_t2w_vols",
        seg_dir="seg_kidney_t2w_fix",
        segs={
            "kv_left": "*left*.nii.gz",
            "kv_right": "*right*.nii.gz",
            "kv_mask": "*mask*.nii.gz",
        },
    ),
    segmentations.KidneyCortexMedullaT2w(
        t2w_seg_dir="seg_kidney_t2w_fix",
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_ax",
        ff_dir="ff_dixon_ax",
        ff_glob="fat_fraction.nii.gz",
        kidney_seg_dir="seg_kidney_dixon_ax",
        kidney_seg_glob="kidney.nii.gz",
        ff_thresh=15,
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_cor",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        kidney_seg_dir="seg_kidney_dixon_cor",
        kidney_seg_glob="kidney.nii.gz",
        ff_thresh=15,
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_totalseg_ax",
        ff_dir="ff_dixon_ax",
        ff_glob="fat_fraction.nii.gz",
        kidney_seg_dir="totalseg_ax",
        kidney_seg_glob="kidneys.nii.gz",
        ff_thresh=15,
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_totalseg_cor",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        kidney_seg_dir="totalseg_cor",
        kidney_seg_glob="kidneys.nii.gz",
        ff_thresh=15,
    ),
    segmentations.KidneyPelvisT2w(
        t2w_seg_dir="seg_kidney_t2w_fix",
        t1_seg_dir="seg_kidney_t1_clean_native",
    ),
    segmentations.KidneyPelvisTrace(
        paren_dir="seg_kidney_t2w_fix",
        whole_dir="seg_kidney_whole",
        whole_glob="kidney_whole.nii.gz",
    ),
    segmentations.OrganFat(
        name="seg_kidney_pelvis_fat_t2w",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        ff_thresh=15,
        seg_dir="seg_kidney_pelvis_t2w",
        seg_glob="kidney_pelvis*.nii.gz",
    ),
    segmentations.OrganFat(
        name="seg_kidney_pelvis_fat_trace",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        ff_thresh=15,
        seg_dir="seg_kidney_pelvis_trace",
        seg_glob="kidney_pelvis*.nii.gz",
    ),
    seg_postprocess.SplitLR(
        srcdir="seg_kidney_cyst_t2w_trace",
        seg_glob="kidney_cyst_fixed.nii.gz",
    ),
    seg_postprocess.SplitLR(
        srcdir="seg_kidney_pelvis_trace",
        seg_glob="whole_kidney_ero.nii.gz",
    ),
    # Statistics and numerical measures
    statistics.Radiomics(
        name="t2w_radiomics",
        deps=["seg_kidney_t2w_fix"],
        params={
            "t2w": {"dir": "../fsort/t2w", "fname": "t2w.nii.gz"},
        },
        segs={
            "tkv_l": {"dir": "seg_kidney_t2w_fix", "fname": "*left*.nii.gz"},
            "tkv_r": {"dir": "seg_kidney_t2w_fix", "fname": "*right*.nii.gz"},
        },
        image_types=["Original", "Exponential"],
        features=["firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"],
    ),
    Stats(),
    StatsDixon(),
    StatsDixonTotalseg(),
    statistics.CMD(
        cmd_params=["t2star", "t2star", "t1", "mtr"],
        skip_params=["t1_noclean"],
    ),
    T1MolliMetadata(),
    statistics.ShapeMetrics(
        name="tkv_shape_metrics",
        seg_dir="seg_kidney_t2w_fix",
        segs={"tkv_l": "*left*.nii.gz", "tkv_r": "*right*.nii.gz"},
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_cor",
        seg_dir="seg_kidney_dixon_cor",
        segs={
            "kidney_left_cor": "kidney_left.nii.gz",
            "kidney_right_cor": "kidney_right.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_ax",
        seg_dir="seg_kidney_dixon_ax",
        segs={
            "kidney_left_ax": "kidney_left.nii.gz",
            "kidney_right_ax": "kidney_right.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_totalseg_cor",
        seg_dir="totalseg_cor",
        segs={
            "kidney_left_cor": "kidney_left.nii.gz",
            "kidney_right_cor": "kidney_right.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="spleen_shape_metrics_totalseg_cor",
        seg_dir="seg_spleen_combined",
        segs={
            "spleen_comb": "spleen.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="liver_shape_metrics_totalseg_cor",
        seg_dir="seg_liver_combined",
        segs={
            "liver_comb": "liver.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="pancreas_shape_metrics_totalseg_cor",
        seg_dir="seg_pancreas_combined",
        segs={
            "pancreas_comb": "pancreas.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_totalseg_ax",
        seg_dir="totalseg_ax",
        segs={
            "kidney_left_ax": "kidney_left.nii.gz",
            "kidney_right_ax": "kidney_right.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.Radiomics(
        name="tkv_radiomics",
        deps=["seg_kidney_t2w_fix"],
        params={
            "t2w": {"dir": "t2w", "fname": "t2w.nii.gz", "src": Module.INPUT},
        },
        segs={
            "tkv_l": {"dir": "seg_kidney_t2w_fix", "fname": "*left*.nii.gz"},
            "tkv_r": {"dir": "seg_kidney_t2w_fix", "fname": "*right*.nii.gz"},
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
    statistics.Radiomics(
        name="kidney_dixon_radiomics_cor",
        deps=["t1_molli_stitch_fix", "seg_kidney_dixon_cor"],
        params={
            "t1": {
                "dir": "t1_molli_stitch_fix",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_l_cor": {
                "dir": "seg_kidney_dixon_cor",
                "fname": "*left*.nii.gz",
            },
            "kidney_dixon_r_cor": {
                "dir": "seg_kidney_dixon_cor",
                "fname": "*right*.nii.gz",
            },
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
    statistics.Radiomics(
        name="kidney_dixon_radiomics_ax",
        deps=["t1_molli_stitch_fix", "seg_kidney_dixon_ax"],
        params={
            "t1": {
                "dir": "t1_molli_stitch_fix",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_l_ax": {
                "dir": "seg_kidney_dixon_ax",
                "fname": "*left*.nii.gz",
            },
            "kidney_dixon_r_ax": {
                "dir": "seg_kidney_dixon_ax",
                "fname": "*right*.nii.gz",
            },
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
    statistics.Radiomics(
        name="kidney_dixon_radiomics_totalseg_cor",
        deps=["t1_molli_stitch_fix", "totalseg_cor"],
        params={
            "t1": {
                "dir": "t1_molli_stitch_fix",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_l_cor": {
                "dir": "totalseg_cor",
                "fname": "*kidney_left*.nii.gz",
            },
            "kidney_dixon_r_cor": {
                "dir": "totalseg_cor",
                "fname": "*kidney_right*.nii.gz",
            },
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
    statistics.Radiomics(
        name="kidney_dixon_radiomics_totalseg_ax",
        deps=["t1_molli_stitch_fix", "totalseg_ax"],
        params={
            "t1": {
                "dir": "t1_molli_stitch_fix",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_l_ax": {
                "dir": "totalseg_ax",
                "fname": "*kidney_left*.nii.gz",
            },
            "kidney_dixon_r_ax": {
                "dir": "totalseg_ax",
                "fname": "*kidney_right*.nii.gz",
            },
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
    statistics.Radiomics(
        name="organ_dixon_radiomics",
        deps=[
            "totalseg_cor",
            "seg_liver_combined",
            "seg_pancreas_combined",
            "seg_spleen_combined",
        ],
        params={
            "water": {
                "dir": "dixon_cor_best",
                "fname": "water.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "liver": {
                "dir": "seg_liver_combined",
                "fname": "liver.nii.gz",
            },
            "pancreas": {
                "dir": "seg_pancreas_combined",
                "fname": "pancreas.nii.gz",
            },
            "spleen": {
                "dir": "seg_spleen_combined",
                "fname": "spleen.nii.gz",
            },
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
    statistics.Radiomics(
        name="t1_radiomics_left",
        deps=["t1_scaled", "t2", "seg_kidney_t1_clean_native"],
        params={
            "t1": {"dir": "t1_scaled", "fname": "t1_conf_scaled_left.nii.gz"},
            "t2_exp": {"dir": "t2", "fname": "t2_exp.nii.gz"},
            "t2_stim": {"dir": "t2", "fname": "t2_stim.nii.gz"},
        },
        segs={
            "cortex_l": {
                "dir": "seg_kidney_t1_clean_native",
                "fname": "*cortex_l*.nii.gz",
            },
            "medulla_l": {
                "dir": "seg_kidney_t1_clean_native",
                "fname": "*medulla_l*.nii.gz",
            },
        },
    ),
    statistics.Radiomics(
        name="t1_radiomics_right",
        deps=["t1_scaled", "t2", "seg_kidney_t1_clean_native"],
        params={
            "t1": {"dir": "t1_scaled", "fname": "t1_conf_scaled_right.nii.gz"},
            "t2_exp": {"dir": "t2", "fname": "t2_exp.nii.gz"},
            "t2_stim": {"dir": "t2", "fname": "t2_stim.nii.gz"},
        },
        segs={
            "cortex_r": {
                "dir": "seg_kidney_t1_clean_native",
                "fname": "*cortex_r*.nii.gz",
            },
            "medulla_r": {
                "dir": "seg_kidney_t1_clean_native",
                "fname": "*medulla_r*.nii.gz",
            },
        },
    ),
    statistics.ShapeMetrics(
        name="wkv_shape_metrics",
        seg_dir="seg_kidney_whole",
        segs={"wkv_l": "*kidney_whole_l.nii.gz", "wkv_r": "*kidney_whole_r.nii.gz"},
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.Radiomics(
        name="wkv_radiomics",
        deps=["t2w", "seg_kidney_whole"],
        params={
            "t2w": {"dir": "t2w", "fname": "t2w.nii.gz", "src": Module.INPUT},
        },
        segs={
            "wkv_l": {"dir": "seg_kidney_whole", "fname": "*kidney_whole_l.nii.gz"},
            "wkv_r": {"dir": "seg_kidney_whole", "fname": "*kidney_whole_r.nii.gz"},
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
    statistics.ShapeMetrics(
        name="wkv_ero_shape_metrics",
        seg_dir="seg_kidney_pelvis_trace_splitlr",
        segs={"wkv_ero_l": "*ero_l*.nii.gz", "wkv_ero_r": "*ero_r*.nii.gz"},
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.Radiomics(
        name="wkv_ero_radiomics",
        deps=["t2w", "seg_kidney_pelvis_trace_splitlr"],
        params={
            "t2w": {"dir": "t2w", "fname": "t2w.nii.gz", "src": Module.INPUT},
        },
        segs={
            "wkv_ero_l": {
                "dir": "seg_kidney_pelvis_trace_splitlr",
                "fname": "*ero_l*.nii.gz",
            },
            "wkv_ero_r": {
                "dir": "seg_kidney_pelvis_trace_splitlr",
                "fname": "*ero_r*.nii.gz",
            },
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
    statistics.ISNR(
        src=Module.INPUT,
        imgs={
            "t1w": "t1w.nii.gz",
            "t2w": "t2w.nii.gz",
        },
    ),
    statistics.KidneyCystStats(
        name="kidney_cyst_stats",
        cyst_dir="seg_kidney_cyst_t2w_clean",
        cyst_glob="kidney_cyst_mask.nii.gz",
        suffix="uon",
    ),
    statistics.KidneyCystStats(
        name="kidney_cyst_stats_trace_fixed",
        cyst_dir="seg_kidney_cyst_t2w_trace",
        cyst_glob="kidney_cyst_fixed.nii.gz",
        suffix="defin",
    ),
    statistics.KidneyCystStats(
        name="kidney_cyst_stats_trace_orig",
        cyst_dir="seg_kidney_cyst_t2w_trace",
        cyst_glob="kidney_cyst_orig.nii.gz",
        suffix="trace",
    ),
    statistics.SegStats(
        name="wkv_volumes",
        segs={
            "wkv_all": {
                "dir": "seg_kidney_whole",
                "glob": "kidney_whole.nii.gz",
            },
            "wkv_l": {
                "dir": "seg_kidney_whole",
                "glob": "kidney_whole_l.nii.gz",
            },
            "wkv_r": {
                "dir": "seg_kidney_whole",
                "glob": "kidney_whole_r.nii.gz",
            },
        },
        seg_volumes=True,
    ),
    statistics.SegStats(
        "wkv_ero_volumes",
        segs={
            "wkv_ero_all": {
                "dir": "seg_kidney_pelvis_trace",
                "glob": "whole_kidney_ero.nii.gz",
            },
            "wkv_ero_all_l": {
                "dir": "seg_kidney_pelvis_trace_splitlr",
                "glob": "whole_kidney_ero_l.nii.gz",
            },
            "wkv_ero_all_r": {
                "dir": "seg_kidney_pelvis_trace_splitlr",
                "glob": "whole_kidney_ero_r.nii.gz",
            },
        },
        seg_volumes=True,
    ),
    statistics.SegStats(
        name="kidney_pelvis_stats_trace",
        segs={
            "kidney_pelvis_l_trace": {
                "dir": "seg_kidney_pelvis_trace",
                "glob": "kidney_pelvis_left.nii.gz",
                "params": [],
            },
            "kidney_pelvis_r_trace": {
                "dir": "seg_kidney_pelvis_trace",
                "glob": "kidney_pelvis_right.nii.gz",
                "params": [],
            },
            "kidney_pelvis_fat_l_trace": {
                "dir": "seg_kidney_pelvis_fat_trace",
                "glob": "kidney_pelvis_left_fat.nii.gz",
                "params": ["ff"],
            },
            "kidney_pelvis_fat_r_trace": {
                "dir": "seg_kidney_pelvis_fat_trace",
                "glob": "kidney_pelvis_right_fat.nii.gz",
                "params": ["ff"],
            },
            "kidney_pelvis_nofat_l_trace": {
                "dir": "seg_kidney_pelvis_fat_trace",
                "glob": "kidney_pelvis_left_nofat.nii.gz",
                "params": [],
            },
            "kidney_pelvis_nofat_r_trace": {
                "dir": "seg_kidney_pelvis_fat_trace",
                "glob": "kidney_pelvis_right_nofat.nii.gz",
                "params": [],
            },
            "kidney_paren_l": {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_left.nii.gz",
                "params": ["ff", "ff_lt25"],
            },
            "kidney_paren_r": {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_right.nii.gz",
                "params": ["ff", "ff_lt25"],
            },
            "kidney_cyst_l": {
                "dir": "seg_kidney_cyst_t2w_trace_splitlr",
                "glob": "*_l.nii.gz",
                "params": [],
            },
            "kidney_cyst_r": {
                "dir": "seg_kidney_cyst_t2w_trace_splitlr",
                "glob": "*_r.nii.gz",
                "params": [],
            },
        },
        params={
            "ff": {
                "dir": "ff_dixon_cor",
                "glob": "fat_fraction.nii.gz",
            },
            "ff_lt25": {
                "dir": "ff_dixon_cor",
                "glob": "fat_fraction.nii.gz",
                "limits": (0, 25),
            },
        },
        seg_volumes=True,
        stats=[
            "n",
            "vol",
            "iqn",
            "iqvol",
            "iqmean",
            "median",
            "iqstd",
            "perc90",
            "te",
            "mode",
            "fwhm",
        ],
    ),
    statistics.SegStats(
        name="kidney_stats_alternate",
        segs={
            "kidney_paren_defin": {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_mask.nii.gz",
                "params": [],
            },
            "kidney_paren_model1": {
                "dir": "seg_kidney_t2w_model1",
                "glob": "kidney_mask.nii.gz",
                "params": [],
            },
            "kidney_paren_model2": {
                "dir": "seg_kidney_t2w",
                "glob": "kidney_mask.nii.gz",
                "params": [],
            },
            "kidney_pelvis_defin": {
                "dir": "seg_kidney_pelvis_trace",
                "glob": "kidney_pelvis.nii.gz",
                "params": [],
            },
            "kidney_pelvis_t2w": {
                "dir": "seg_kidney_pelvis_t2w",
                "glob": "kidney_pelvis.nii.gz",
                "params": [],
            },
        },
        seg_volumes=True,
    ),
    statistics.SegVolumeDiffs(
        name="cyst_volume_diffs",
        stats_files=[
            "kidney_cyst_stats/kidney_cyst.csv",
            "kidney_cyst_stats_trace_fixed/kidney_cyst.csv",
            "kidney_cyst_stats_trace_orig/kidney_cyst.csv",
        ],
        diffs={
            "cyst_def_min_trace": ("kidney_cyst_defin_vol", "kidney_cyst_trace_vol"),
            "cyst_def_min_uon": ("kidney_cyst_defin_vol", "kidney_cyst_uon_vol"),
            "cyst_trace_min_uon": ("kidney_cyst_trace_vol", "kidney_cyst_uon_vol"),
        },
    ),
    statistics.SegStats(
        name="organ_volumes",
        segs={
            "liver_local": {
                "dir": "seg_liver_dixon_ax",
                "glob": "kidney_whole.nii.gz",
            },
            "liver_totalseg_ax": {
                "dir": "totalseg_ax",
                "glob": "liver.nii.gz",
            },
            "liver_totalseg_cor": {
                "dir": "totalseg_cor",
                "glob": "liver.nii.gz",
            },
            "liver_comb": {
                "dir": "seg_liver_combined",
                "glob": "liver.nii.gz",
            },
            "pancreas_local": {
                "dir": "seg_pancreas_ethrive",
                "glob": "pancreas.nii.gz",
            },
            "pancreas_totalseg_ax": {
                "dir": "totalseg_ax",
                "glob": "pancreas.nii.gz",
            },
            "pancreas_totalseg_cor": {
                "dir": "totalseg_cor",
                "glob": "pancreas.nii.gz",
            },
            "pancreas_comb": {
                "dir": "seg_pancreas_combined",
                "glob": "pancreas.nii.gz",
            },
            "spleen_local": {
                "dir": "seg_spleen_dixon_ax",
                "glob": "spleen.nii.gz",
            },
            "spleen_totalseg_ax": {
                "dir": "totalseg_ax",
                "glob": "spleen.nii.gz",
            },
            "spleen_totalseg_cor": {
                "dir": "totalseg_cor",
                "glob": "spleen.nii.gz",
            },
            "sat_local": {
                "dir": "seg_sat_dixon_ax",
                "glob": "sat.nii.gz",
            },
            "sat_totalseg_ax": {
                "dir": "totalseg_ax",
                "glob": "subcutaneous_fat.nii.gz",
            },
            "sat_totalseg_cor": {
                "dir": "totalseg_cor",
                "glob": "subcutaneous_fat.nii.gz",
            },
            "vat_cor_local": {
                "dir": "seg_vat_dixon_cor_local",
                "glob": "vat.nii.gz",
            },
            "vat_cor_totalseg": {
                "dir": "seg_vat_dixon_cor_totalseg",
                "glob": "vat.nii.gz",
            },
            "vat_ax_local": {
                "dir": "seg_vat_dixon_ax_local",
                "glob": "vat.nii.gz",
            },
            "vat_ax_totalseg": {
                "dir": "seg_vat_dixon_ax_totalseg",
                "glob": "vat.nii.gz",
            },
        },
        seg_volumes=True,
    ),
    # TempAddPelvis(),
]


def add_options(parser):
    parser.add_argument(
        "--t1-scale-factors", help="Directory containing manual fixed cyst masks"
    )
    parser.add_argument(
        "--cyst-masks", help="Directory containing manual fixed cyst masks"
    )
    parser.add_argument(
        "--seg-kidney-t2w-fix", help="Directory containing manual T2w kidney masks"
    )
    parser.add_argument(
        "--seg-kidney-t1-fix",
        help="Directory containing manual kidney cortex/medulla masks + maps",
    )
    parser.add_argument(
        "--seg-kidney-cyst-trace", help="Directory containing TRACE cyst segmentations"
    )
    parser.add_argument(
        "--seg-organs-trace", help="Directory containing TRACE organ segmentations"
    )
