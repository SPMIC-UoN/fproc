from afirm import *

NAME = "pluto"
STUDYDIR = "/spmstore/project/RenalMRI/pluto/"
OUTNAME = NAME

BATCH = {
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
        "nnunetv2-img",
    ],
    "time": "24:00:00",
    "mem_per_cpu": "64g",
    "subjects": 25,
    "extra_fproc_args": [
        "--cyst-masks", os.path.join(STUDYDIR, "fixed_cyst_masks"),
        "--kidney-t1-model=/software/imaging/body_pipelines/trained_models//kidney_t1_molli_min_max.pt",
        "--kidney-t2w-model=/software/imaging/body_pipelines/trained_models//t2w_seg.h5",
        "--seg-kidney-t2w-fix", os.path.join(STUDYDIR, "fixed_masks_new"),
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
        "fproc/stats_dixon/stats.csv",
        "fproc/kidney_dixon_shape_metrics_ax/shape_metrics.csv",
        "fproc/kidney_dixon_shape_metrics_cor/shape_metrics.csv",
        "fproc/kidney_dixon_radiomics_ax/radiomics.csv",
        "fproc/kidney_dixon_radiomics_cor/radiomics.csv",
    ],
    "shape": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/seg_kidney_t2w_vols/volumes.csv",
        "fproc/tkv_shape_metrics/shape_metrics.csv",
        "fproc/tkv_radiomics/radiomics.csv",
    ],
}
