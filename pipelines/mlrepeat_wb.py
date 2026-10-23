import os

from rapamune import *

# Configuration
NAME = "mlrepeat_wb"
STUDYDIR = "/gpfs01/spmstore/project/RenalMRI/mlrepeat"
OUTNAME = "mlrepeat"

BATCH = {
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
        "conda-img",
        "nnunetv2-img",
        "ukbbseg-img",
        "totalseg-img",
        "leg-seg-dixon-img",
        "leg-dixon-seg-img",
        "musclemap-img",
    ],
    "time": "12:00:00",
    "gpu": True,
    "mem_per_cpu": "32g",
    "subjects": 150,
    "input_subfolder": "",
    "output_subfolder": "",
}

COHORTS = [
    ("full cohort", "", os.path.join(STUDYDIR, "subjects_to_report.txt")),
]
