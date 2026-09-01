import os

from fproc.modules import regrid
from chain import *

# Configuration
NAME = "rapamune"
STUDYDIR = os.path.join("/gpfs01/spmstore/project/RenalMRI", NAME)
OUTNAME = NAME

COHORTS = [
    ("full cohort", "", os.path.join(STUDYDIR, "subjects_to_report.txt")),
]

class NormalizeRawDixon(Module):
    def __init__(self, name="raw_dixon_norm", **kwargs):
        Module.__init__(self, name, deps=["raw_dixon_singlevols"], **kwargs)

    def process(self):
        dir = self.kwargs.get("dir", "raw_dixon_singlevols")
        src = self.kwargs.get("src", Module.OUTPUT)
        spec = self.kwargs.get("spec", {})
        for glob, factor in spec.items():
            for img in self.inimgs(dir, glob, src=src):
                if factor is None:
                    LOG.info(f" - {img.fname}: no normalization")
                    img_data = img.data
                elif isinstance(factor, str):
                    value = getattr(img, factor, 1.0)
                    LOG.info(f" - {img.fname}: normalization factor = {factor}: {value}")
                    img_data = img.data / value
                elif isinstance(factor, (int, float)):
                    LOG.info(f" - {img.fname}: normalization factor = {factor}")
                    img_data = img.data / factor
                elif isinstance(factor, (list, tuple)):
                    img_data = img.data
                    for f in factor:
                        if isinstance(f, str):
                            value = getattr(img, f, 1.0)
                            LOG.info(f" - {img.fname}: normalization factor = {f}: {value}")
                            img_data /= value
                        elif isinstance(f, (int, float)):
                            LOG.info(f" - {img.fname}: normalization factor = {f}")
                            img_data /= f
                        else:
                            self.bad_data("Invalid factor type: {}".format(type(f)))    
                else:
                    self.bad_data("Invalid factor type: {}".format(type(factor)))

                img.save_derived(img_data, self.outfile(img.fname))


# Tweak pipeline to rescale dixon images
MODULES.insert(1, NormalizeRawDixon(
    spec = {
        "raw_dixon_series_?_1.nii.gz": None,
        "raw_dixon_series_?_2.nii.gz": ["PhilipsRWVSlope", "PhilipsRWVSlope"],
        "raw_dixon_series_?_3.nii.gz": None,
        "raw_dixon_series_?_4.nii.gz": None,
        "raw_dixon_series_?.nii.gz": None,
    }
))
MODULES[2] = regrid.Stitch(
        name="dixon_stitched",
        img_dir="raw_dixon_norm",
        imgs={
            "raw_dixon_series_?_1.nii.gz": "raw_dixon_1.nii.gz",
            "raw_dixon_series_?_2.nii.gz": "raw_dixon_2.nii.gz",
            "raw_dixon_series_?_3.nii.gz": "raw_dixon_3.nii.gz",
            "raw_dixon_series_?.nii.gz": "raw_dixon.nii.gz",
        },
        normalise=False
    )
