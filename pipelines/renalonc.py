# Duplicate of AFIRM pipeline for RENALONC

import logging
import os

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

from afirm import *

# Configuration
NAME = "renalonc"
STUDYDIR = os.path.join("/gpfs01/spmstore/project/RenalMRI", NAME)
OUTNAME = NAME

COHORTS = [
    ("full cohort", "", None),
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
        "fproc/seg_kidney_t2w_vols_nocyst/volumes.csv",
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
