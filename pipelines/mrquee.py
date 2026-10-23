# MRQUEE: runs the shared demistifi_local_data_3t pipeline with study-specific postrun config
import os

from demistifi_local_data_3t import MODULES, add_options, __version__

# Configuration
NAME = "mrquee"
STUDYDIR = "/gpfs01/spmstore/project/RenalMRI/demistifi_local_data_3t"
OUTNAME = NAME

BATCH = {
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
        "nnunetv2-img",
        "ukbbseg-img",
        "totalseg-img",
    ],
    "time": "06:00:00",
    "mem_per_cpu": "32g",
    "subjects": 30,
    "input_subfolder": "",
    "extra_fproc_args" : [
        "--add-niftis",
        os.path.join(STUDYDIR, "data", NAME),
        "--totalseg-fixes",
        os.path.join(STUDYDIR, "qformcorrected_t1maps2"),
    ]
}

COHORTS = [
    ("full cohort", "", None),
]

OUTFILES = {
    "": [
        "fproc/stats/stats.csv",
        "fproc/liver_radiomics/radiomics.csv",
        "fproc/spleen_radiomics/radiomics.csv",
        "fproc/pancreas_radiomics/radiomics.csv",
        "fproc/lung_radiomics/radiomics.csv",
        "fproc/organ_radiomics/radiomics.csv",
        "fproc/shape_metrics/shape_metrics.csv",
    ],
    "totalseg": [
        "fproc/totalseg/volumes.csv",
        "fproc/totalseg/volumes_dilated.csv",
    ],
}
