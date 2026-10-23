from afirm import *

# Configuration
NAME = "dlo"
STUDYDIR = "/gpfs01/spmstore/project/RenalMRI/demistifi_local_addons"
OUTNAME = NAME

BATCH = {
    "time": "48:00:00",
    "mem_per_cpu": "32g",
    "subjects": 100,
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
        "nnunetv2-img",
        "totalseg-img",
        "traceseg-img",
    ],
    "extra_fproc_args": [
        "--cyst-masks",
        "/spmstore/project/RenalMRI/demistifi_local_addons//fixed_cyst_masks",
        "--kidney-t1-model=/software/imaging/body_pipelines/trained_models//kidney_t1_molli_min_max.pt",
        "--kidney-t2w-model=/software/imaging/body_pipelines/trained_models//kidney_t2w_afirm_platnum_v02.h5",
        "--seg-kidney-t2w-fix=/spmstore/project/RenalMRI/demistifi_local_addons//DEFINITIVETKVMASKS_RENAMED",
        "--seg-kidney-cyst-trace=/spmstore/project/RenalMRI/demistifi_local_addons//TRACE_DATA_RENAMED",
        "--seg-organs-trace=/spmstore/project/RenalMRI/demistifi_local_addons//TRACE_ORGANS_RENAMED",
    ],
}

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
