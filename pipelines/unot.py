# UNOT: runs the shared demistifi_local_data_3t pipeline with study-specific postrun config

from demistifi_local_data_3t import MODULES, add_options, __version__

# Configuration
NAME = "unot"
STUDYDIR = "/gpfs01/spmstore/project/RenalMRI/demistifi_local_data_3t"
OUTNAME = NAME

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
