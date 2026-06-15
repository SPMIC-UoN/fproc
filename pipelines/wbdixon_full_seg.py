# Segmentation step of full wbdixon pipeline from XNAT dicom data
import logging

import numpy as np

from fproc.module import Module
from fproc.modules import maps, statistics, regrid, segmentations, seg_postprocess

LOG = logging.getLogger(__name__)

__version__ = "0.0.1"

NAME = "wbdixon_full"


class RoiZ(Module):
    def __init__(self, name="roiz", **kwargs):
        self._src_dir = kwargs.get("src_dir", "dixon")
        Module.__init__(self, name, deps=[self._src_dir], **kwargs)

    def process(self):
        src_glob = self.kwargs.get("src_glob", "*.nii.gz")
        imgs = self.inimgs(self._src_dir, src_glob)
        prop = float(self.kwargs.get("proportion", 50)) / 100

        for img in imgs:
            LOG.info(f" - Processing {img.fname}")
            img = img.reorient2std()
            img_data = np.copy(img.data)
            slice_idx = int(img.shape[2] * prop)
            LOG.info(f" - Z dim {img_data.shape[2]} zeroing from slice {slice_idx}")
            img_data[:, :, slice_idx:, ...] = 0
            img.save_derived(img_data, self.outfile(img.fname))


MODULES = [
    maps.FatFractionDixon(dixon_dir="dixon_classify", ff_name="fat_fraction"),
    maps.T2starDixon(dixon_dir="dixon_classify", t2star_name="t2star"),
    RoiZ(name="dixon_classify_roiz", src_dir="dixon_classify", proportion=66),
    # Segmentations
    segmentations.LegDixon(dixon_dir="dixon_classify_roiz"),
    segmentations.TotalSeg(src_dir="dixon_classify", img_glob="water.nii.gz"),
]
