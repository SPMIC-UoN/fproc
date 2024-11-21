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
    def __init__(self, seg_dir, name=None, **kwargs):
        if name is None:
            name = seg_dir + "_fix"
        Module.__init__(self, name, **kwargs)
        self._seg_dir = seg_dir

    def process(self):
        fix_dir_option = self.kwargs.get("fix_dir_option", self.name + "_fixed_dir")
        fixed_segs = None
        fix_dir = getattr(self.pipeline.options, fix_dir_option, None)
        try_to_fix = False
        if not fix_dir:
            LOG.info(" - No fixed segmentations dir specified")
        elif not os.path.exists(fix_dir):
            LOG.info(f" - Fixed seg dir {fix_dir} does not exist")
        else:
            try_to_fix = True

        map_dir = self.kwargs.get("map_dir", None)
        map_fname = self.kwargs.get("map_fname", None)
        if map_dir and map_fname:
            map_img = self.single_inimg(map_dir, map_fname)
        else:
            map_img = None

        segs = self.kwargs.get("segs", {})
        for seg_glob, fix_glob in segs.items():
            seg_img = self.single_inimg(self._seg_dir, seg_glob, src=self.kwargs.get("seg_src", self.OUTPUT))
            if seg_img is None:
                LOG.warn(f"No segmentation found matching {self._seg_dir}/{seg_glob} - ignoring")
                continue

            fixed = False
            if try_to_fix:
                LOG.info(f" - Checking {seg_img.fname}")
                globexpr = os.path.join(fix_dir, fix_glob % self.pipeline.options.subjid)
                fixed_segs = glob.glob(globexpr)
                if not fixed_segs:
                    LOG.info(f" - No fixed segmentation found")
                    break
                if len(fixed_segs) > 1:
                    LOG.warn(f" - Multiple matching 'fixed' segmentations found: {fixed_segs} - using first")
                fixed_img = ImageFile(fixed_segs[0])
                LOG.info(f" - Saving fixed segmentation from {fixed_img.fname}")
                fixed = True

            if not fixed:
                LOG.info(f" - Saving original segmentation from {seg_img.fname}")
                fixed_img = seg_img

            fixed_img.save(self.outfile(seg_img.fname))
            if map_img:
                self.lightbox(map_img, fixed_img, name=f"{fixed_img.fname}_lightbox", tight=True)

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

            # Find matching T1 map and resample T2w mask for this segmentation
            t1_map = self.matching_img(t1_seg, t1_maps)
            if t1_map is None:
                LOG.warn(f" - Could not find matching T1 map for {t1_seg.fname} - no overlay will be generated")
            else:
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

class LargestBlob(Module):
    def __init__(self, srcdir, seg_glob, overlay_srcdir=None, overlay_glob=None):
        Module.__init__(self, f"{srcdir}_largestblob")
        self._srcdir = srcdir
        self._seg_glob = seg_glob
        self._overlay_srcdir = overlay_srcdir
        self._overlay_glob = overlay_glob

    def process(self):
        segs = self.inimgs(self._srcdir, self._seg_glob, src=self.OUTPUT)
        for seg in segs:
            out_fname = self.outfile(seg.fname)
            blobs = self.blobs_by_size(seg.data)
            if not blobs:
                LOG.warn(f"No mask for {seg.fname}")
                seg.save(out_fname)
                continue

            largest = blobs[0]
            LOG.info(f" - Largest blob for {seg.fname} has {np.count_nonzero(largest)} voxels")
            LOG.info(f" - Also found {len(blobs) - 1} other blobs")

            seg.save_derived(largest, out_fname)
            LOG.info(f" - Saving to {out_fname}")

class SegVolumes(Module):
    def __init__(self, name="seg_volumes", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        seg_dir = self.kwargs.get("seg_dir", "seg_kidney_t2w")
        segs = self.kwargs.get("segs", {})
        vol_fname = self.kwargs.get("vol_fname", "volumes.csv")

        LOG.info(f" - Saving volumes from {seg_dir} to {vol_fname}")
        with open(self.outfile(vol_fname), "w") as f:
            for name, seg in segs.items():
                seg_img = self.single_inimg(seg_dir, seg, src=self.OUTPUT)
                if seg_img is None:
                    LOG.warn(f"Segmentation not found: {seg}")
                else:
                    f.write("%s,%.2f\n" % (name, seg_img.voxel_volume * np.count_nonzero(seg_img.data)))

class KidneyCystClean(Module):
    def __init__(self, name="seg_kidney_cyst_t2w_clean", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t2w_glob = self.kwargs.get("seg_t2w_glob", "*mask*.nii.gz")
        t2w_dir = self.kwargs.get("seg_t2w_dir", "seg_kidney_t2w")
        t2w_src = self.kwargs.get("seg_t2w_src", self.OUTPUT)
        t2w_mask = self.single_inimg(t2w_dir, t2w_glob, src=t2w_src)

        cyst_glob = self.kwargs.get("cyst_glob", "kidney_cyst_mask.nii.gz")
        cyst_dir = self.kwargs.get("cyst_dir", "seg_kidney_cyst_t2w")
        cyst_src = self.kwargs.get("cyst_src", self.OUTPUT)
        cyst_seg = self.single_inimg(cyst_dir, cyst_glob, src=cyst_src)
        if cyst_seg is None:
            self.no_data("No T2w kidney cyst segmentation found to clean")

        if t2w_mask is not None:
            LOG.info(f" - Cleaning {cyst_seg.fname} using T2w mask {t2w_mask.fname}")
            t2w_mask_res_nii = self.resample(t2w_mask, cyst_seg, is_roi=True, allow_rotated=True)
            t2w_mask_res = (t2w_mask_res_nii.get_fdata() > 0).astype(np.int8)
            cyst_seg.save_derived(t2w_mask_res, self.outfile("t2w_mask_res.nii.gz"))
            combined_mask = (t2w_mask_res + cyst_seg.data > 0).astype(np.int8)
            cyst_seg.save_derived(combined_mask, self.outfile("combined_mask.nii.gz"))
            blobs = self.blobs_by_size(combined_mask)
            if len(blobs) < 2:
                LOG.warn(f"Only {len(blobs)} blobs found - expected 2 (one for each kidney)")
            else:
                kidney_blobs = blobs[:2]
            kidney_blobs = (sum(kidney_blobs) > 0).astype(np.int8)
            cyst_seg.save_derived(kidney_blobs, self.outfile("kidney_blobs.nii.gz"))
            cleaned_data = kidney_blobs
            cleaned_data[t2w_mask_res > 0] = 0
        else:
            LOG.warn("No T2w kidney segmentation found - will not clean kidney cyst segmentation")
            cleaned_data = cyst_seg.data

        cyst_seg.save_derived(cleaned_data, self.outfile("kidney_cyst_mask.nii.gz"))
        cleaned_mask = self.inimg(self.name, "kidney_cyst_mask.nii.gz", src=self.OUTPUT)

        t2w_map = self.single_inimg(cyst_dir, "kidney_cyst_0000.nii.gz", src=cyst_src)
        if t2w_map is not None:
            self.lightbox(t2w_map, cleaned_mask, name="kidney_cyst_t2w_lightbox", tight=True) 
        else:
            LOG.warn("No T2w map found - will not create lightbox image")

        # Count number of cysts and volume
        total_volume = np.count_nonzero(cleaned_mask.data) * cleaned_mask.voxel_volume
        labelled = skimage.measure.label(cleaned_mask.data)
        props = skimage.measure.regionprops(labelled)
        num_cysts = len(props)
        with open(self.outfile("kidney_cyst.csv"), "w") as f:
            f.write(f"vol_kidney_cyst,{total_volume}\n")
            f.write(f"count_kidney_cysts,{num_cysts}\n")

