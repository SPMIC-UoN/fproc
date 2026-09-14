import os

from fproc.modules import regrid
from amulet import *

# Configuration
NAME = "selina"
STUDYDIR = os.path.join("/gpfs01/spmstore/project/LiverMRI", NAME)
OUTNAME = NAME

COHORTS = [
    ("full cohort", "", None),
]
