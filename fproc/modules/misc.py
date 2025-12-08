"""
FPROC: Miscellaneous modules
"""
import logging
import os
import sys

import fsl.wrappers as fsl
import nibabel as nib
import numpy as np
import scipy

from fproc.module import Module

LOG = logging.getLogger(__name__)

class ScanDates(Module):
    """
    Output scan date and DOB from DICOM metadata if present
    """
    def __init__(self, name="scan_dates", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        input = self.kwargs.get("input", None)
        if not input:
            self.no_data("No input directories specified")

        scan_date = None
        for in_dir, in_glob in input.items():
            imgs = self.inimgs(in_dir, in_glob, src=self.OUTPUT)
            if imgs:
                scan_date = imgs[0].acquisitiondatetime
                if scan_date:
                    LOG.info(f" - Found scan date: {scan_date} in {imgs[0].fname}" )
                    break
        
        if not scan_date:
            LOG.info(f" - No images with scan date found")
            scan_date = ""

        dob = None
        for in_dir, in_glob in input.items():
            imgs = self.inimgs(in_dir, in_glob, src=self.OUTPUT)
            if imgs:
                dob = imgs[0].dob
                if dob:
                    LOG.info(f" - Found DOB: {dob} in {imgs[0].fname}" )
                    break
        
        if not dob:
            LOG.info(f" - No images with DOB found")
            dob = ""
        if isinstance(dob, list):
            dob = dob[0]

        scan_date = scan_date.lower().split("t")[0]
        dob = dob.lower().split("t")[0]
        scan_key = self.kwargs.get("scan_key", "scan_date")
        dob_key = self.kwargs.get("dob_key", "dob")
        with open(self.outfile(f"{self.name}.csv"), "w") as f:
            f.write(f"{scan_key}," + scan_date + "\n")
            f.write(f"{dob_key}," + dob + "\n")
