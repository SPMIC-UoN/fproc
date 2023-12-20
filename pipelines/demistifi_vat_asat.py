import logging
import os

import numpy as np
import skimage

from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class VAT(Module):
    """
    Internal fat from DIXON
    """
    def __init__(self):
        Module.__init__(self, "vat")

    def process(self):
        """
        VAT (internal fat)

        i. Abdominal cavity mask - organ masks (liver, spleen, kidneys, lungs, pancreas) can we use pancreas T1w seg regridded onto Dixon?
        ii. Threshold the whole body fat percent image (include >0.9) and fill small holes 
        iii. Final VAT mask = ixii
        """
        abd_cavity = self.inimg("qpdata", "seg_abdominal_cavity_dixon.nii.gz")
        if not abd_cavity:
            self.no_data("No abdominal cavity image found")

        spleen = self.inimg("qpdata", "seg_spleen_dixon.nii.gz", warn=True, check=False)
        kidney_right = self.inimg("qpdata", "seg_kidney_right_dixon.nii.gz", warn=True, check=False)
        kidney_left = self.inimg("qpdata", "seg_kidney_right_dixon.nii.gz", warn=True, check=False)
        lungs = self.inimg("qpdata", "seg_lungs_dixon.nii.gz", warn=True, check=False)
        liver = self.inimg("qpdata", "seg_liver_dixon.nii.gz", warn=True, check=False)
        pancreas = self.inimg("qpdata", "seg_pancreas_t1w.nii.gz", warn=True, check=False)
        if pancreas is not None:
            pancreas = self.resample(pancreas, abd_cavity, is_roi=False, allow_rotated=True)

        organs = None
        for arr in (spleen.data, kidney_right.data, kidney_left.data, lungs.data, liver.data, pancreas.get_fdata()):
            if arr is None:
                LOG.warn("Could not remove organ data")
                continue
            arr = (arr > 0).astype(np.int32)
            if organs is None:
                organs = np.copy(arr)
            else:
                organs += arr
        organs = (organs > 0).astype(np.int32)
        no_organs = (abd_cavity.data > 0).astype(np.int32) - organs
        no_organs[no_organs < 0] = 0

        fat = self.inimgs("preproc", "*/analysis/fat.percent.nii.gz")
        if not fat:
            self.no_data("No body fat percent image found")
        elif len(fat) > 1:
            self.bad_data("Multiple body fat percent images found")
        fat = (fat[0].data > 0.9).astype(np.int32)

        vat = (no_organs * fat).astype(np.int32)
        abd_cavity.save_derived(vat, self.outfile("vat_nofill.nii.gz"))

        # Fill small holes (1 voxel) slicewise in Z direction
        for z in range(vat.shape[2]):
            inv_vat_slice = np.logical_not(vat[..., z]).astype(np.int32)
            labelled = skimage.measure.label(inv_vat_slice)
            props = skimage.measure.regionprops(labelled)
            for region in props:
                if np.sum(inv_vat_slice[labelled == region.label]) < 2:
                    vat[..., z][labelled == region.label] = 1

        abd_cavity.save_derived(vat, self.outfile("vat.nii.gz"))

        water = self.inimgs("preproc", "*/nifti/water.nii.gz")
        self.lightbox(water[0], vat, "vat_lightbox", tight=True)

class ASAT(Module):
    def __init__(self):
        Module.__init__(self, "asat")

    def process(self):
        """
        ASAT (Abdominal subcutaneous adipose tissue = external tissue)
        
        i. Invert Body cavity mask
        ii. Threshold the whole body fat percent image (include >0.9) and fill small holes
        ii. Bounding box =  upper and lower bounds of the abdominal mask in the superior-inferior (z) direction.
        iv. Final ASAT mask = 1x2x3
        """
        body_mask = self.inimgs("preproc", "*/analysis/mask.body.nii.gz")
        if not body_mask:
            self.no_data("No body mask image found")
        elif len(body_mask) > 1:
            self.bad_data("Multiple body mask images found")

        body_cavity = self.inimg("qpdata", "seg_body_cavity_dixon.nii.gz")
        asat = (body_mask[0].data > 0).astype(np.int32) - (body_cavity.data > 0).astype(np.int32)
        asat[asat < 0] = 0

        fat = self.inimgs("preproc", "*/analysis/fat.percent.nii.gz")
        if not fat:
            self.no_data("No body fat percent image found")
        elif len(fat) > 1:
            self.bad_data("Multiple body fat percent images found")
        fat = (fat[0].data > 0.9).astype(np.int32)
        asat = asat * fat

        abd_cavity = self.inimg("qpdata", "seg_abdominal_cavity_dixon.nii.gz")
        nonzero_slices_in_abd_cavity = [z for z in range(abd_cavity.shape[2]) if np.count_nonzero(abd_cavity.data[..., z]) > 0]
        bb_bottom, bb_top = min(nonzero_slices_in_abd_cavity), max(nonzero_slices_in_abd_cavity)
        LOG.info(f"Abdominal cavity bounding box in Z direction: {bb_bottom} to {bb_top}")

        asat[..., bb_top+1:] = 0
        asat[..., :bb_bottom] = 0
        abd_cavity.save_derived(asat, self.outfile("asat.nii.gz"))
        
        water = self.inimgs("preproc", "*/nifti/water.nii.gz")
        self.lightbox(water[0], asat, "asat_lightbox", tight=True)

MODULES = [
    VAT(),
    ASAT(),
]

class DemistifiPostprocArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "demistifi_postproc", __version__)
        
class DemistifiPostproc(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "demistifi_postproc", __version__, DemistifiPostprocArgumentParser(), MODULES)

if __name__ == "__main__":
    DemistifiPostproc().run()
