import os

from fproc.modules import regrid
from rapamune import *

# Configuration
NAME = "mlrepeat_wb"
STUDYDIR = "/gpfs01/spmstore/project/RenalMRI/mlrepeat"
OUTNAME = "mlrepeat"

COHORTS = [
    ("full cohort", "", os.path.join(STUDYDIR, "subjects_to_report.txt")),
]
