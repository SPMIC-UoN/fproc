"""
FPROC: Modules for aligning data
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

class FlirtAlignOnly(Module):
    """
    Use flirt to align data but don't change the data space of the data being moved
    """
    def __init__(self, in_dir, in_glob, ref_dir, ref_glob, name=None, **kwargs):
        if not name:
            name = f"{in_dir}_to_{ref_dir}_flirt"
        Module.__init__(self, name, **kwargs)
        self._in_dir = in_dir
        self._in_glob = in_glob
        self._ref_dir = ref_dir
        self._ref_glob = ref_glob

    def process(self):
        imgs_to_align = self.inimgs(self._in_dir, self._in_glob, src=self.OUTPUT)
        run_flirt = True
        if not imgs_to_align:
            LOG.warn(f" - No image found matching {self._in_dir}/{self._in_glob} - nothing to align")
            run_flirt = False

        ref_img = self.single_inimg(self._ref_dir, self._ref_glob, src=self.OUTPUT)
        if ref_img is None:
            LOG.warn(f" - No alignment will be performed - assuming images already aligned")
            run_flirt = False
        else:
            LOG.info(f" - Reference image: {ref_img.fname}")

        weight_mask = None
        if run_flirt and "weight_mask" in self.kwargs and "weight_mask_dir" in self.kwargs:
            weight_mask = self.single_inimg(self.kwargs["weight_mask_dir"], self.kwargs["weight_mask"], src=self.OUTPUT)
            if weight_mask is None:
                LOG.warn(" - No weight mask found - will not be able to use as input weighting")
            else:
                LOG.info(f" - Weight mask image: {weight_mask.fname}")
                weight_mask_data = weight_mask.data
                for idx in range(self.kwargs.get("weight_mask_dil", 0)):           
                    weight_mask_data = scipy.ndimage.binary_dilation(weight_mask_data, structure=np.ones((5, 5, 5)))
                weight_mask = weight_mask.save_derived(weight_mask_data, self.outfile("weight_mask.nii.gz"))

        for img in imgs_to_align:
            align_fname = self.outfile(img.fname)
            if run_flirt:
                LOG.info(f" - Aligning {img.fname}")
                ref_img_res = self.resample(ref_img, img, is_roi=False, allow_rotated=True)
                
                flirt_opts = {
                    "schedule" : os.path.join(os.environ["FSLDIR"], "etc", "flirtsch", "sch3Dtrans_3dof"),
                    "bins" : 256,
                    "cost" : "corratio",
                    "interp" : "trilinear",
                    "log" : {
                        "stderr" : sys.stdout,
                        "stdout" : sys.stdout,
                        "cmd" : sys.stdout,
                    }
                }
                
                if weight_mask is not None:
                    weight_fname = self.outfile(f"{img.fname_noext}_weight_mask.nii.gz")
                    weight_mask_res = self.resample(weight_mask, img, is_roi=True, allow_rotated=True)
                    weight_mask_res.to_filename(weight_fname)
                    flirt_opts["inweight"] = weight_fname
                else:
                    weight_mask_res = None

                if True or any([d == 1 for d in img.shape[:3]]) == 1:
                    LOG.info(" - Aligning single slice data - using 2D registration")
                    flirt_opts["twod"] = True
                    flirt_opts["paddingsize"] = 1       
                    flirt_opts["schedule"] = os.path.join(os.environ["FSLDIR"], "etc", "flirtsch", "sch2D_3dof"),

                flirt_result = fsl.flirt(img.nii, ref_img_res, out=fsl.LOAD, omat=fsl.LOAD, **flirt_opts)
                img_align = flirt_result["out"]
                mat = flirt_result["omat"]
                LOG.info(f" - Transform:\n{mat}")
            else:
                img_align = img.nii
                mat = None
            LOG.info(f" - Saving {align_fname}")
            img_align.to_filename(align_fname)

            for also_dir, also_glob in self.kwargs.get("also_align", {}).items():
                extra_imgs = self.inimgs(also_dir, also_glob, src=self.OUTPUT)
                for extra_img in extra_imgs:
                    if extra_img.affine_matches(img):
                        align_fname = self.outfile(extra_img.fname)
                        if mat is not None:
                            LOG.info(f" - Applying to: {extra_img.fname}")
                            apply_result = fsl.applyxfm(extra_img.nii, ref_img_res, mat=mat, interp="trilinear", paddingsize=1, out=fsl.LOAD)
                            img_align = apply_result["out"]
                            if "kidney" in extra_img.fname or "seg" in extra_img.fname:
                                # FIXME below is only for segmentations...
                                img_align = nib.Nifti1Image(img_align.get_fdata() > 0.5, img_align.header.get_best_affine(), img_align.header)
                        else:
                            img_align = extra_img.nii
                        LOG.info(f" - Saving {align_fname}")
                        img_align.to_filename(align_fname)
        else:
            # If no maps just save extra images as they are
            for also_dir, also_glob in self.kwargs.get("also_align", {}).items():
                extra_imgs = self.inimgs(also_dir, also_glob, src=self.OUTPUT)
                for extra_img in extra_imgs:
                    align_fname = self.outfile(extra_img.fname)
                    LOG.info(f" - Saving {align_fname}")
                    extra_img.save(align_fname)
