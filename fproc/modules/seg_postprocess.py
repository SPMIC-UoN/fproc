"""
FPROC: Modules for post-processing segmentations
"""
import glob
import logging
import os

import numpy as np
import skimage
import scipy

from fsort import ImageFile
from fproc.module import Module

LOG = logging.getLogger(__name__)

class SegFix(Module):
    """
    Module that can replace automatic segmentations with a manually drawn one
    """
    def __init__(self, srcdir, src_fname, seg_srcdir, seg_src_fname, fix_dir_option, fix_glob=None):
        Module.__init__(self, f"{srcdir}_fix")
        self.srcdir = srcdir
        self.src_fname = src_fname
        self.seg_srcdir = seg_srcdir
        self.seg_src_fname = seg_src_fname
        self.fix_dir_option = fix_dir_option
        self.fix_glob = fix_glob
        if fix_glob:
            self.fix_glob = fix_glob
        else:
            self.fix_glob = "%s_*.nii.gz"

    def process(self):
        origs = self.inimgs(self.srcdir, self.src_fname, is_depfile=True)
        if not origs:
            orig = None
        else:
            if len(origs) > 1:
                LOG.warn(f" - Multiple files found matching {self.src_fname}- using first")
            orig = origs[0]

        fix_dir = getattr(self.pipeline.options, self.fix_dir_option, None)
        if not fix_dir:
            LOG.info(" - No fixed files dir specified")
            new = None
        else:
            globexpr = os.path.join(fix_dir, self.fix_glob % self.pipeline.options.subjid)
            news = glob.glob(globexpr)
            if not news:
                LOG.info(f" - No fixed file for {self.pipeline.options.subjid} in {globexpr}")
                new = None
            else:
                if len(news) > 1:
                    LOG.warn(f" - Multiple fixed files found for {self.pipeline.options.subjid}: {news} - using first")
                new = ImageFile(news[0])

        if new is not None:
            LOG.info(f" - Saving fixed file from {new.fname}")
        elif orig is not None:
            LOG.info(f" - Using original file from {orig.fname}")
            new = orig
        else:
            LOG.warn(f" - No original or fixed file found")

        if new is not None:        
            new.save(self.outfile(self.src_fname))

            # Overlay onto source img
            seg_src_map = self.inimgs(self.seg_srcdir, self.seg_src_fname)
            if not seg_src_map:
                LOG.warn(f" - Could not find source map for for overlay: {self.seg_srcdir}, {self.seg_src_fname}")
            else:
                self.lightbox(seg_src_map[0], new, name=f"{self.srcdir}_lightbox", tight=True)

class KidneyT1Clean(Module):
    def __init__(self, srcdir="seg_kidney_t1", seg_t2w_srcdir="seg_kidney_t2w", t1_map_srcdir="t1_kidney", generic=True, t2w=True):
        Module.__init__(self, "seg_kidney_t1_clean")
        self._seg_t1_srcdir = srcdir
        self._seg_t2w_srcdir = seg_t2w_srcdir
        self._t1_map_srcdir = t1_map_srcdir
        self._generic = generic
        self._t2w = t2w

    def process(self):
        t1_segs = self.inimgs(self._seg_t1_srcdir, "kidney_*.nii.gz", src=self.OUTPUT)
        t1_maps = self.inimgs(self._t1_map_srcdir, "t1_map*.nii.gz", src=self.OUTPUT)
        t2w_masks = self.inimgs(self._seg_t2w_srcdir, "kidney_mask.nii.gz", src=self.OUTPUT)
        if not t1_segs:
            self.no_data(" - No T1 segmentations found to clean")
        if not t2w_masks:
            t2w_masks = [None]
        elif len(t2w_masks) > 1 and self._t2w:
            LOG.warn(f" - Multiple T2w segmentations - using first: {t2w_masks[0].fname}")
        t2w_mask = t2w_masks[0]

        for t1_seg in t1_segs:
            # Find matching T1 map and resample T2w mask for this segmentation
            t1_map = self.matching_img(t1_seg, t1_maps)
            if t1_map is None:
                LOG.warn(f" - Could not find matching T1 map for {t1_seg.fname} - ignoring this mask")
                continue

            if self._t2w:
                if t2w_mask is not None:
                    LOG.info(f" - Cleaning {t1_seg.fname} using T2w mask {t2w_mask.fname}")
                    t2w_mask_res = self.resample(t2w_mask, t1_seg, is_roi=True, allow_rotated=True)
                    cleaned_data_t1_seg = self._clean_t2w(t1_seg.data, t2w_mask_res.get_fdata())
                elif self._generic:
                    LOG.warn(f" - Could not find T2w mask for {t1_seg.fname} - using generic cleaning")
                    cleaned_data_t1_seg = self._clean_generic(t1_seg, t1_segs)
                else:
                    LOG.warn(f" - Could not find T2w mask for {t1_seg.fname} - not cleaning")
                    cleaned_data_t1_seg = t1_seg
            elif self._generic:
                LOG.info(f" - Cleaning {t1_seg.fname} using generic algorithm")
                cleaned_data_t1_seg = self._clean_generic(t1_seg, t1_segs)
            else:
                LOG.warn(f"No cleaning algorithms specified - not cleaning")
                cleaned_data_t1_seg = t1_seg

            t1_seg.save_derived(cleaned_data_t1_seg, self.outfile(t1_seg.fname))
            self.lightbox(t1_map.data, cleaned_data_t1_seg, f"{t1_seg.fname_noext}_t1_cleaned_lightbox")

    def _clean_generic(self, t1_seg, t1_segs, remove_small=True):
        # How close to the edge (as a fraction of total pixels) a blob centroid needs to be
        # before it is discarded
        EDGE_VERT_FRACTION = 0.1
        EDGE_HORIZ_FRACTION = 0.1

        # How close to the horizontal centre (as a fraction of total pixels) a blob centroid needs to be
        # before it is discarded (the kidneys should be either side of the central spine)
        CENTRE_FRACTION = 1.0/12

        # Fraction of pixels to use as criteria for small blob removal. The minimum size of a blob
        # is this fraction of the horizontal dimension squared
        SMALL_FRACTION = 1.0/20

        # Get the whole kidney mask matching the segmentation we are cleaning
        mask_img_cor, mask_img_med = None, None
        for seg in t1_segs:
            if seg.fname_noext.endswith("medulla_t1") and seg.affine_matches(t1_seg):
                mask_img_med = seg.data
            elif seg.fname_noext.endswith("cortex_t1") and seg.affine_matches(t1_seg):
                mask_img_cor = seg.data

        if mask_img_cor is None or mask_img_med is None:
            self.bad_data(f"Could not find cortex and medulla mask images matching {t1_seg.fname}")

        mask_img_cor[mask_img_cor<0] = 0
        mask_img_cor[mask_img_cor>0] = 1
        mask_img_med[mask_img_med<0] = 0
        mask_img_med[mask_img_med>0] = 1
        kid_mask = np.logical_or(mask_img_cor, mask_img_med)
        cleaned_data = np.copy(t1_seg.data)

        for slice_idx in range(kid_mask.shape[2]):
            kid_mask_slice = kid_mask[..., slice_idx]
            labelled = skimage.measure.label(kid_mask_slice)
            props = skimage.measure.regionprops(labelled)

            # Remove any central blobs
            for region in props:
                if (region.centroid[0] < kid_mask.shape[0]*(0.5+CENTRE_FRACTION) and 
                    region.centroid[0] > kid_mask.shape[0]*(0.5-CENTRE_FRACTION)):
                    kid_mask_slice[labelled == region.label] = 0

            # Remove any blobs around the edge
            for region in props:
                if (region.centroid[1] < kid_mask.shape[0]*EDGE_HORIZ_FRACTION or
                    region.centroid[1] > kid_mask.shape[0]*(1-EDGE_HORIZ_FRACTION) or
                    region.centroid[0] < kid_mask.shape[1]*EDGE_VERT_FRACTION or
                    region.centroid[0] > kid_mask.shape[1]*(1-EDGE_VERT_FRACTION)):
                    kid_mask_slice[labelled == region.label] = 0

            # Remove any small blobs from one copy
            if remove_small:
                smallblob_thresh = round((kid_mask_slice.shape[0]*SMALL_FRACTION)**2)
                for region in props:
                    if np.sum(kid_mask_slice[labelled == region.label]) < smallblob_thresh:
                        kid_mask_slice[labelled == region.label] = 0

            cleaned_data[..., slice_idx] *= kid_mask_slice

        return cleaned_data.astype(np.uint8)

    def _clean_t2w(self, t1_seg, t2w_mask):
        # Dilate T2w masks by 2 voxels
        mask_data_dil = scipy.ndimage.binary_dilation(t2w_mask, structure=np.ones((3, 3, 3)))
        mask_data_dil = scipy.ndimage.binary_dilation(mask_data_dil, structure=np.ones((3, 3, 3)))
        cleaned_data = (t1_seg * mask_data_dil).astype(np.uint8)
        LOG.debug(f" - Voxel counts: orig {np.count_nonzero(t1_seg)}, mask {np.count_nonzero(t2w_mask)}, dil mask {np.count_nonzero(mask_data_dil)}, out {np.count_nonzero(cleaned_data)},")
        return cleaned_data
