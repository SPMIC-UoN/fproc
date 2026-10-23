# ACLAIM: runs the shared resus pipeline with study-specific postrun config

import os

from resus import MODULES, add_options, __version__

# Configuration
NAME = "aclaim"
STUDYDIR = os.path.join("/gpfs01/spmstore/project/RenalMRI", NAME)
OUTNAME = NAME

BATCH = {
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
        "nnunetv2-img",
        "ukbbseg-img",
        "totalseg-img",
    ],
    "time": "01:00:00",
    "mem_per_cpu": "16g",
    "subjects": 10,
    "input_subfolder": "",
    "extra_fproc_args": [
        "--kidney-t2w-model=/software/imaging/body_pipelines/trained_models//t2w_seg.h5",
        "--add-niftis", os.path.join(STUDYDIR, "add_maps"),
        "--pancreas-masks", os.path.join(STUDYDIR, "pancreas_masks"),
        "--liver-masks", os.path.join(STUDYDIR, "liver_masks"),
    ],
}

COHORTS = [
    ("full cohort", "", None),
]

OUTFILES = {
    "": [
        "fproc/stats/stats.csv",
        "fproc/radiomics/radiomics.csv",
        "fproc/lung_radiomics/radiomics.csv",
    ],
    "totalseg": [
        "fproc/totalseg/volumes.csv",
        "fproc/totalseg/volumes_dilated.csv",
        "fproc/shape_metrics_totalseg/radiomics.csv",
    ],
}
