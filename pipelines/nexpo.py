import logging
import os

from fproc.module import Module
from fproc.modules import statistics
from fsort.image_file import ImageFile

LOG = logging.getLogger(__name__)

__version__ = "0.0.1"

NAME = "nexpo"


class RoiData(Module):
    def __init__(self, name="roi_data", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        indir = self.pipeline.options.input
        roifile = os.path.normpath(
            os.path.join(indir, os.pardir, self.pipeline.options.subjid + ".nii.gz")
        )
        LOG.info(f" - Looking for ROIs in {roifile}")
        if not os.path.exists(roifile):
            self.no_data("No ROI file found")
        img = ImageFile(roifile, warn_json=False)
        img.save(self.outfile("rois.nii.gz"))
        LOG.info(" - Saved as rois.nii.gz")
        roi1 = img.data == 1
        img.save_derived(roi1.astype("uint8"), self.outfile("roi1.nii.gz"))
        roi2 = img.data == 2
        img.save_derived(roi2.astype("uint8"), self.outfile("roi2.nii.gz"))
        LOG.info(" - Saved ROI1 as roi1.nii.gz and ROI2 as roi2.nii.gz")


MODULES = [
    RoiData(),
    statistics.Radiomics(
        name="radiomics_shape",
        params={
            "roi_data": {"dir": "roi_data", "fname": "rois.nii.gz"},
        },
        segs={
            "roi1": {"dir": "roi_data", "fname": "roi1.nii.gz"},
            "roi2": {"dir": "roi_data", "fname": "roi2.nii.gz"},
        },
        features=["shape"],
    ),
]
