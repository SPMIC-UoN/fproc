# ACLAIM: runs the shared resus pipeline with study-specific postrun config

import os

from resus import MODULES, add_options, __version__

# Configuration
NAME = "aclaim"
STUDYDIR = os.path.join("/gpfs01/spmstore/project/RenalMRI", NAME)
OUTNAME = NAME

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
