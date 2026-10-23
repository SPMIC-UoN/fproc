from afirm import *

NAME = "tk21"
STUDYDIR = "/gpfs01/spmstore/project/RenalMRI/tk21"
OUTNAME = NAME

BATCH = {
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
        "conda-img",
        "nnunetv2-img",
    ],
    "time": "24:00:00",
    "mem_per_cpu": "32g",
    "subjects": 50,
    "extra_fproc_args": [
        "--kidney-t1-model=/software/imaging/body_pipelines/trained_models//kidney_t1_molli_min_max.pt",
        "--kidney-t2w-model=/software/imaging/body_pipelines/trained_models//t2w_seg.h5",
        "--seg-kidney-t1-fix", os.path.join(STUDYDIR, "seg_kidney_t1_fix",
   ],
}

COHORTS = [
    ("full cohort", "", None),
]

OUTFILES = {
    "kidney": [
        "fproc/stats/stats.csv",
        "fproc/t1_molli_md/t1_molli_md.csv",
    ],
    "t1_radiomics_left" : [
        "fproc/t1_radiomics_left/radiomics.csv",
    ],
    "t1_radiomics_right" : [
        "fproc/t1_radiomics_right/radiomics.csv",
    ],
    "organs" : [
        "fproc/stats_dixon/stats.csv",
        "fproc/kidney_dixon_shape_metrics_ax/shape_metrics.csv",
        "fproc/kidney_dixon_shape_metrics_cor/shape_metrics.csv",
        "fproc/kidney_dixon_radiomics_ax/radiomics.csv",
        "fproc/kidney_dixon_radiomics_cor/radiomics.csv",
    ],
    "shape" : [
        "fproc/seg_kidney_t2w_vols/volumes.csv",
        "fproc/tkv_shape_metrics/shape_metrics.csv",
        "fproc/tkv_radiomics/radiomics.csv",
    ],
}
