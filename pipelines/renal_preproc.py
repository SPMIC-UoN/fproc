import glob
import logging
import os
import shutil
import sys
import validators
import wget

from fsort.image_file import ImageFile

from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, CopyModule
import fproc.stats

import numpy as np
import scipy
import skimage
import nibabel as nib
import fsl.wrappers as fsl
from dbdicom.wrappers.skimage import _volume_features

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class T1(Module):
    def __init__(self):
        Module.__init__(self, "t1")

    def process(self):
        rawmollis = self.inimgs("t1", "t1_raw_molli*.nii.gz")
        if rawmollis:
            LOG.info(f" - Found raw MOLLI data - performing T1 map reconstruction")
            from ukat.mapping.t1 import T1
            # FIXME temp until we can extract from DICOM
            inversion_times = [235, 315, 1235, 1315, 2235, 2315, 3235, 4235]
            for img in rawmollis:
                slice_timing = img.SliceTiming if img.SliceTiming is not None else 0
                LOG.info(f" - Reconstructing {img.fname} using TIs {inversion_times} and slice timing {slice_timing}")
                mapper = T1(img.data, inversion_times, img.affine, np.array(slice_timing))
                img.save_derived(mapper.t1_map, self.outfile(img.fname.replace("t1_raw_molli", "t1_map")))
                img.save_derived(mapper.t1_map, self.outfile(img.fname.replace("t1_raw_molli", "t1_conf")))
        else:
            self.copyinput("t1", "t1_map*.nii.gz")
            self.copyinput("t1", "t1_conf*.nii.gz")

        t1maps = self.inimgs("t1", "t1_map*.nii.gz", is_depfile=True)
        for t1map in t1maps:
            LOG.info(f" - Segmenting T1 map from {t1map.fname}")
            self.runcmd([
                'kidney_t1_seg',
                '--input', t1map.dirname,
                '--subjid', '',
                '--display-id', self.pipeline.options.subjid,
                '--t1', t1map.fname,
                '--model', self.pipeline.options.t1_model,
                '--noclean',
                '--output', self.outdir,
                '--outprefix', f'seg_kidney_{t1map.fname_noext}'],
                logfile=f'seg_kidney_{t1map.fname_noext}.log'
            )

class B0(Module):
    def __init__(self):
        Module.__init__(self, "b0")

    def process(self):
        phasedata, tes = [], []
        phasemaps = self.inimgs("b0", "b0_phase_echo*.nii.gz")
        if not phasemaps:
            realmaps = self.inimgs("b0", "b0_real_echo*.nii.gz")
            imagmaps = self.inimgs("b0", "b0_imag_echo*.nii.gz")
            if not realmaps:
                self.no_data("No phase or real part B0 maps")
            if not imagmaps:
                self.no_data("No phase or imaginary part B0 maps")
            if len(realmaps) != len(imagmaps):
                self.bad_data("Different number of real and imaginary maps")

            for real, imag in zip(realmaps, imagmaps):
                if real.echotime != imag.echotime:
                    self.bad_data(f"Real and imaginary maps {real.fname}, {imag.fname} do not have the same echo time: {real.echotime} vs {imag.echotime}")
                LOG.info(f" - Calculating phase from real/imag: {real.fname}, {imag.fname}, TE: {real.echotime}")
                data_phase = -np.arctan2(imag.data, real.data)
                phasedata.append(data_phase)
                real.save_derived(real.data, self.outfile(real.fname.replace("_real", "_realcopy")))
                real.save_derived(imag.data, self.outfile(real.fname.replace("_real", "_imagcopy")))
                real.save_derived(data_phase, self.outfile(real.fname.replace("_real", "_phase")))
                tes.append(real.echotime * 1000)
                srcfile = real
        else:
            for f in phasemaps:
                LOG.info(f" - Found phase data: {f.fname}, TE: {f.echotime}")
                phasedata.append(f.data)
                tes.append(f.echotime * 1000)
                srcfile = f

        if len(phasedata) != 2:
            LOG.warn(" - More than two echos found - using first two only")
            phasedata = phasedata[:2]
            tes = tes[:2]

        stacked_data = np.stack(phasedata, axis=-1)
        from ukat.mapping.b0 import B0
        mapper = B0(stacked_data, tes, affine=srcfile.affine)

        # Save output maps to Nifti
        srcfile.save_derived(mapper.b0_map, self.outfile("b0.nii.gz"))
        srcfile.save_derived(mapper.phase0, self.outfile("b0phase0.nii.gz"))
        srcfile.save_derived(mapper.phase1, self.outfile("b0phase1.nii.gz"))
        srcfile.save_derived(mapper.phase_difference, self.outfile("b0phasediff.nii.gz"))

class MTR(Module):
    def __init__(self):
        Module.__init__(self, "mtr")

    def process(self):
        ondata = self.inimg("mtr", "mtr_on.nii.gz", check=False)
        offdata = self.inimg("mtr", "mtr_off.nii.gz", check=False)
        if ondata is not None and offdata is not None:
            LOG.info(f" - Using MTR ON/OFF data from {ondata.fname}, {offdata.fname}")
            off_on_data = np.stack([offdata.data, ondata.data], axis=-1)
            srcfile = ondata
        else:
            onoffdata = self.inimg("mtr", "mtr_on_off.nii.gz", check=False)
            if onoffdata is None:
                self.no_data("No MTR on/off data found")
            LOG.info(f" - Using MTR ON/OFF data from {onoffdata.fname}")
            off_on_data = np.flip(onoffdata.data, axis=-1)
            srcfile = onoffdata

        from ukat.mapping.mtr import MTR
        mapper = MTR(off_on_data, affine=srcfile.affine)
        srcfile.save_derived(mapper.mtr_map, self.outfile("mtr.nii.gz"))

class T2star(Module):
    def __init__(self):
        Module.__init__(self, "t2star")

    def process(self):
        echos = self.inimgs("t2star", "t2star_e_*.nii.gz")
        if len(echos) != 12:
            self.bad_data(f"Expected 12 echos, got {len(echos)}")

        echos.sort(key=lambda x: x.EchoTime)
        imgdata = [e.data for e in echos]
        tes = np.array([1000*e.EchoTime for e in echos])
        LOG.info(f"TEs: {tes}")
        last_echo = echos[-1]
        affine = last_echo.affine
        last_echo.save_derived(last_echo.data, self.outfile("last_echo.nii.gz"))  

        voxel_sizes = last_echo.nii.header.get_zooms()
        resample_voxel_size = self.pipeline.options.t2star_resample
        if resample_voxel_size > 0:
            voxel_sizes = [round(v, 1) for v in voxel_sizes][:2]
            std_sizes = [round(resample_voxel_size, 1) for v in voxel_sizes][:2]
            if not np.allclose(voxel_sizes, std_sizes):
                LOG.info(f"T2* data has resolution: {voxel_sizes} - resampling to {std_sizes}")
                zoom_factors = [voxel_sizes[d] / std_sizes[d] for d in range(2)] + [1.0] * (last_echo.ndim - 2)
                imgdata = [scipy.ndimage.zoom(d, zoom_factors) for d in imgdata]
                # Need to scale affine by same amount so FOV is unchanged. Note that this means the corner
                # voxels will have the same co-ordinates in original and rescaled data. The FOV depicted
                # in fsleyes etc will be slightly smaller because the voxels are smaller and the co-ordinates
                # are defined as being the centre of the voxel
                for j in range(2):
                    affine[:, j] = affine[:, j] / zoom_factors[j]

        methods = [self.pipeline.options.t2star_method]
        if methods == ["all"]:
            methods = ["loglin", "2p_exp"]
        for method in methods:
            from ukat.mapping.t2star import T2Star
            mapper = T2Star(np.stack(imgdata, axis=-1), tes, affine=affine, method=method, multithread=False)
            last_echo.save_derived(mapper.t2star_map, self.outfile(f"t2star_{method}.nii.gz"))
            last_echo.save_derived(self._fix_r2star_units(mapper.r2star_map()), self.outfile(f"r2star_{method}.nii.gz"))
            last_echo.save_derived(mapper.m0_map, self.outfile(f"m0_{method}.nii.gz"))

            LOG.info(f" - Generating T1 segmentation overlay images for {method} T2* map")
            t1segs = self.inimgs("t1", "seg_kidney_*.nii.gz", is_depfile=True)
            for seg in t1segs:
                res_seg = self.resample(seg, last_echo, is_roi=True)
                seg_data = res_seg.get_fdata()
                seg_name = seg.fname_noext.strip("seg_kidney_")
                LOG.debug(f" - Generating overlay images for segmentation image {seg_name}")
                self.lightbox(mapper.t2star_map, seg_data, f"t2star_{seg_name}_lightbox")

    def _fix_r2star_units(self, r2star_data):    
        """
        Change R2* from ms^-1 to s^-1
        
        FIXME do we need to check current range?
        """
        return 1000.0*r2star_data

class B1(CopyModule):
    def __init__(self):
        CopyModule.__init__(self, "b1")

class T1w(Module):
    def __init__(self):
        Module.__init__(self, "t1w")

    def process(self):
        img = self.inimg("t1w", "t1w.nii.gz")
        img.save_derived(img.data, self.outfile("t1w.nii.gz"))

        LOG.info(f" - Generating T1 segmentation overlay images for T1w map")
        t1segs = self.inimgs("t1", "seg_kidney_*.nii.gz", is_depfile=True)
        for seg in t1segs:
            res_seg = self.resample(seg, img, is_roi=True)
            seg_data = res_seg.get_fdata()
            seg_name = seg.fname_noext.strip("seg_kidney_")
            LOG.debug(f" - Generating overlay images for segmentation image {seg_name}")
            self.lightbox(img.data, seg_data, f"t1w_{seg_name}_lightbox")

class T2w(Module):
    def __init__(self):
        Module.__init__(self, "t2w")

    def process(self):
        model_weights = self.pipeline.options.t2w_model
        if validators.url(model_weights):
            # Download weights from supplied URL
            # FIXME check MD5 hash to validate?
            LOG.info(f" - Downloading model weights from {model_weights}")
            wget.download(model_weights, "model.h5")
            model_weights = "model.h5"
        else:
            LOG.info(f" - Using model weights from {model_weights}")

        # The segmentor needs both the image array and affine so the size of each voxel is known. post_process=True removes all but
        # the largest two areas in the mask e.g. removes small areas of incorrectly categorised tissue. This can cause issues if the
        # subject has more or less than two kidneys though.
        img = self.inimg("t2w", "t2w.nii.gz")
        img.save_derived(img.data, "t2w.nii.gz")
    
        LOG.info(f" - Generating T2w kidney segmentation from {img.fname}")
        from ukat.segmentation import whole_kidney
        segmentation = whole_kidney.Segmentation(img.data, img.affine, post_process=True, binary=True, weights=model_weights)
        segmentation.to_nifti(output_directory=self.outdir, base_file_name=f"seg_t2w", maps=['mask', 'left', 'right', 'individual'])

        # Check for 'fixed' masks and use these if available
        mask_fname = self.outfile("seg_t2w_mask.nii.gz")
        mask_orig_fname = self.outfile("seg_t2w_mask_orig.nii.gz")

        # Check for fixed masks and replace
        used_fixed = False
        fixed_masks_dir = self.pipeline.options.t2w_fixed_masks
        if fixed_masks_dir:
            LOG.info(f" - Checking for fixed T2w masks in {fixed_masks_dir}/{self.pipeline.options.subjid}")
            fix_subjdir = os.path.join(fixed_masks_dir, self.pipeline.options.subjid)
            if os.path.isdir(fix_subjdir):
                LOG.debug(f" - Found fixed masks dir: {fix_subjdir}")
                for root, dirs, files in os.walk(fix_subjdir):
                    for fname in files:
                        if "fix" in fname.lower():
                            LOG.debug(f" - Found fixed mask {fname}")
                            if "mask_fix" in fname.lower():
                                LOG.info(f" - Replacing T2w mask {mask_orig_fname} with fixed file: {fname}")
                                shutil.copyfile(mask_fname, mask_orig_fname)
                                shutil.copyfile(os.path.join(root, fname), mask_fname)

                                LOG.info(f" - Splitting fixed T2w mask into left/right kidneys")
                                mask_img = ImageFile(mask_fname)
                                mask_data = mask_img.data
                                LOG.debug(f" - Affine:\n{mask_img.affine}")
                                axcodes = nib.orientations.aff2axcodes(mask_img.affine)
                                LOG.debug(f" - Axis codes: {axcodes}")
                                lr_axis = axcodes.index("L") if "L" in axcodes else axcodes.index("R")
                                lr_centre = mask_data.shape[lr_axis] // 2
                                l_data, r_data = np.copy(mask_data), np.copy(mask_data)
                                lr_slices = [slice(None)] * 3

                                lr_slices[lr_axis] = slice(lr_centre) if "L" in axcodes else slice(lr_centre, mask_data.shape[lr_axis])
                                LOG.debug(f" - Slices: {lr_slices}")
                                l_data[tuple(lr_slices)] = 0
                                fname_left = mask_fname.replace("_mask.nii", "_left_kidney.nii")
                                shutil.copyfile(fname_left, fname_left.replace(".nii", "_orig.nii"))
                                mask_img.save_derived(l_data, fname_left)

                                lr_slices[lr_axis] = slice(lr_centre) if "R" in axcodes else slice(lr_centre, mask_data.shape[lr_axis])
                                LOG.debug(f" - Slices: {lr_slices}")
                                r_data[tuple(lr_slices)] = 0
                                fname_right = mask_fname.replace("_mask.nii", "_right_kidney.nii")
                                shutil.copyfile(fname_right, fname_right.replace(".nii", "_orig.nii"))
                                mask_img.save_derived(r_data, fname_right)
                            
                                used_fixed = True
                            else:
                                LOG.warn(f" - Found 'fixed' mask {fname} but does not match naming convention - ignoring")
        if not used_fixed:
            LOG.info(f" - No 'fixed' mask found - using original")
            mask_img = ImageFile(mask_fname)

        # Sometimes the Nifti masks come out with in64 data type which breaks the XNAT viewer - fix this
        # FIXME find cleaner way to do this
        LOG.info(f" - Correcting data types for segmentation files")
        for fname in glob.glob(os.path.join(self.outdir, "seg_t2w*.nii.gz")):
            LOG.debug(f" - Correcting data type for {fname}")
            seg_img = ImageFile(fname)
            seg_data = seg_img.data.astype(np.uint8)
            seg_img.save_derived(seg_data, fname)

        # Generate overlay images using T2w map
        t2w_map = self.inimg("t2w", "t2w.nii.gz")
        LOG.info(f" - Generating overlay image for T2w segmentation using {t2w_map.fname}")
        self.lightbox(t2w_map, mask_img, "seg_t2w_lightbox")

        LOG.info(f" - Saving volumes to tkv.csv")
        img_total = ImageFile(self.outfile("seg_t2w_mask.nii.gz"))
        img_left = ImageFile(self.outfile("seg_t2w_left_kidney.nii.gz"))
        img_right = ImageFile(self.outfile("seg_t2w_right_kidney.nii.gz"))
        voxel_vol = img_total.voxel_volume
        with open(self.outfile("kidney_volumes.csv"), "w") as f:
            f.write("kv_left,%.2f\n" % (voxel_vol * np.count_nonzero(img_left.data)))
            f.write("kv_right,%.2f\n" % (voxel_vol * np.count_nonzero(img_right.data)))
            f.write("kv_total,%.2f\n" % (voxel_vol * np.count_nonzero(img_total.data)))
            f.write("kv_fixed_mask,%i\n" % int(used_fixed))

class T1Clean(Module):
    def __init__(self):
        Module.__init__(self, "t1_clean")

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
        mask_data_dil = scipy.ndimage.binary_dilation(t2w_mask.data, structure=np.ones((3, 3, 3)))
        mask_data_dil = scipy.ndimage.binary_dilation(mask_data_dil, structure=np.ones((3, 3, 3)))
        cleaned_data = (t1_seg.data * mask_data_dil).astype(np.uint8)
        LOG.debug(f" - Voxel counts: orig {np.count_nonzero(t1_seg.data)}, mask {np.count_nonzero(t2w_mask.data)}, dil mask {np.count_nonzero(mask_data_dil)}, out {np.count_nonzero(cleaned_data)},")
        return cleaned_data

    def process(self):
        t1_segs = self.inimgs("align", "seg_kidney_*.nii.gz", is_depfile=True)
        t1_maps = self.inimgs("align", "t1_map*.nii.gz", is_depfile=True)
        if not t1_segs:
            self.no_data(" - No T1 segmentations found to clean")

        t2w_masks = self.inimgs("resample", "seg_t2w_mask_res_t1_map*.nii.gz", is_depfile=True)

        for t1_seg in t1_segs:
            # Find matching T1 map and resample T2w mask for this segmentation
            t1_map = self.matching_img(t1_seg, t1_maps)
            if t1_map is None:
                continue

            cleaned_basename = t1_seg.fname_noext + "_cleaned"
            t2w_mask = self.matching_img(t1_seg, t2w_masks)
            if t2w_mask is None:
                LOG.warn(f" - Could not find matching T2w mask for {t1_seg.fname} - using generic cleaning")
                cleaned_data_t1_seg = self._clean_generic(t1_seg, t1_segs)
            else:
                LOG.info(f" - Cleaning T1 segmentation {t1_seg.fname} using T2w mask {t2w_mask.fname}")
                cleaned_data_t1_seg = self._clean_t2w(t1_seg, t2w_mask)

            t1_seg.save_derived(cleaned_data_t1_seg, self.outfile(cleaned_basename + ".nii.gz"))

            # Generate overlay images using T1 map
            self.lightbox(t1_map.data, cleaned_data_t1_seg, cleaned_basename + "_lightbox")

class Resample(Module):
    def __init__(self):
        Module.__init__(self, "resample")

    def process(self):
        t1_maps = self.inimgs("align", "t1_map*.nii.gz", is_depfile=True)
        if not t1_maps:
            self.no_data("No T1 maps found - will not resample T2w/T1w onto T1 grid")

        t2w_segs = self.inimgs("t2w", "seg_t2w*.nii.gz", is_depfile=True)
        if not t2w_segs:
            LOG.warn(" - No T2w segmentations found - will not resample onto T1 grid")

        for t1_map in t1_maps:
            for t2w_seg in t2w_segs:
                LOG.info(f" - Resampling {t2w_seg.fname} onto T1 map grid {t1_map.fname}")
                nii_res = self.resample(t2w_seg, t1_map, is_roi=True)
                t1_map.save_derived(nii_res.get_fdata(), self.outfile(t2w_seg.fname_noext + "_res_" + t1_map.fname_noext + ".nii.gz"))

        t1w_map = self.inimg("t1w", "t1w.nii.gz", check=False, warn=True)
        if t1w_map is not None:
            LOG.info(f" - Resampling {t1w_map.fname} onto T1 map grid {t1_map.fname}")
            nii_res = self.resample(t1w_map, t1_map, is_roi=False)
            t1_map.save_derived(nii_res.get_fdata(), self.outfile(t1w_map.fname_noext + "_res_" + t1_map.fname_noext + ".nii.gz"))

class Alignment(Module):
    def __init__(self):
        Module.__init__(self, "align")

    def process(self):
        no_align = self.pipeline.options.t1_no_realign

        t1_confs = self.inimgs("t1", "t1_conf*.nii.gz", is_depfile=True)
        if not t1_confs:
            LOG.warn("No T1 confidence maps found - will not align to T2*")
            no_align = True

        t2star_lastecho = self.inimg("t2star", "last_echo.nii.gz", is_depfile=True, check=False)
        if t2star_lastecho is None:
            LOG.warn(" - No T2* last echo found - will not align to T1")
            no_align = True

        if not no_align:
            t2w_mask = self.inimg("t2w", "seg_t2w_mask.nii.gz", is_depfile=True, check=False)
            if t2w_mask is None:
                LOG.warn(" - No T2w kidney mask found - will not be able to use as input weighting")
            else:
                t2w_mask_data_dil = scipy.ndimage.binary_dilation(t2w_mask.data, structure=np.ones((5, 5, 5)))
                t2w_mask_data_dil = scipy.ndimage.binary_dilation(t2w_mask_data_dil, structure=np.ones((5, 5, 5)))
                t2w_mask_data_dil = scipy.ndimage.binary_dilation(t2w_mask_data_dil, structure=np.ones((5, 5, 5)))
                t2w_mask_data_dil = scipy.ndimage.binary_dilation(t2w_mask_data_dil, structure=np.ones((5, 5, 5)))
                t2w_mask_data_dil = scipy.ndimage.binary_dilation(t2w_mask_data_dil, structure=np.ones((5, 5, 5)))
                t2w_mask_data_dil = scipy.ndimage.binary_dilation(t2w_mask_data_dil, structure=np.ones((5, 5, 5)))
                t2w_mask.save_derived(t2w_mask_data_dil, self.outfile("t2w_mask_dil.nii.gz"))
                t2w_mask_dil = ImageFile(self.outfile("t2w_mask_dil.nii.gz"))

        for t1_conf in t1_confs:
            t1_map = ImageFile(t1_conf.fpath.replace("t1_conf", "t1_map"))
            t1_align_fname = self.outfile(t1_map.fname_noext + "_align_t2star.nii.gz")
            t1_conf_align_fname = self.outfile(t1_conf.fname_noext + "_align_t2star.nii.gz")

            if not no_align:
                LOG.info(f" - Aligning T1 map {t1_map.fname} to T2* {t2star_lastecho.fname}")

                t2star_res_nii = self.resample(t2star_lastecho, t1_map, is_roi=False, allow_rotated=True)
                flirt_opts = {
                    "schedule" : os.path.join(os.environ["FSLDIR"], "etc", "flirtsch", "sch3Dtrans_3dof"),
                    "bins" : 256,
                    "cost" : "corratio",
                    #"searchrx" : "-20 20",
                    #"searchry" : "-20 20",
                    #"searchrz" : "-20 20",
                    "interp" : "trilinear",
                }
                if self.pipeline.options.debug:
                    flirt_opts["log"] = {
                        "stderr" : sys.stdout,
                        "stdout" : sys.stdout,
                        "cmd" : sys.stdout,
                    }

                if t2w_mask is not None:
                    LOG.info(" - Setting T2w kidney mask as input weight")
                    t2w_mask_res_nii = self.resample(t2w_mask_dil, t1_map, is_roi=True, allow_rotated=True)
                    t2w_mask_res_nii.to_filename(self.outfile("inweight_temp.nii.gz"))
                    flirt_opts["inweight"] = self.outfile("inweight_temp.nii.gz")

                if any([d == 1 for d in t1_map.shape[:3]]) == 1:
                    LOG.info(" - T1 is single slice - using 2D mode")
                    flirt_opts["twod"] = True
                    flirt_opts["paddingsize"] = 1       
                    flirt_opts["schedule"] = os.path.join(os.environ["FSLDIR"], "etc", "flirtsch", "sch2D_3dof"),

                flirt_result = fsl.flirt(t1_map.nii, t2star_res_nii, out=fsl.LOAD, omat=fsl.LOAD, **flirt_opts)
                img = flirt_result["out"]
                mat = flirt_result["omat"]
                LOG.debug(f" - Transform:\n{mat}")

                LOG.info(f" - Saving result as {t1_align_fname}")
                img.to_filename(t1_align_fname)

                LOG.info(f" - Applying to T1 confidence map: {t1_conf.fname} -> {t1_conf_align_fname}")
                apply_result = fsl.applyxfm(t1_conf.nii, t2star_res_nii, mat=mat, interp="trilinear", paddingsize=1, out=fsl.LOAD)
                img = apply_result["out"]
                img.to_filename(t1_conf_align_fname)
                LOG.info(f" - Applying to T1 segmentations")
            else:
                LOG.info(f" - No alignment - keeping original T1 maps {t1_map.fname} , {t1_conf.fname}")
                t1_map.nii.to_filename(t1_align_fname)
                t1_conf.nii.to_filename(t1_conf_align_fname)

            t1_segs = self.inimgs("t1", "seg_*.nii.gz", is_depfile=True)
            for t1_seg in t1_segs:
                align_fname = self.outfile(t1_seg.fname_noext + "_align_t2star.nii.gz")
                if t1_seg.affine_matches(t1_map):
                    if not no_align:
                        LOG.debug(f" - Applying to T1 segmentation: {t1_seg.fname} -> {align_fname}")
                        apply_result = fsl.applyxfm(t1_seg.nii, t2star_res_nii, mat=mat, interp="trilinear", paddingsize=1, out=fsl.LOAD)
                        img = apply_result["out"]
                        t1_seg.save_derived(img.get_fdata() > 0.5, align_fname)
                    else:
                        t1_seg.save_derived(t1_seg.data, align_fname)

class SegStats(Module):
    def __init__(self):
        Module.__init__(self, "stats")

        # How to handle multiple param/seg combinations with non zero output. 
        # 'combine' means take all overlapping data, 'best' means choose combination
        # with most overlap
        self.multi_mode = "combine"

        # Mapping from name to a glob which matches segmentation files
        # If the glob matches multiple files, the resulting stats will be handled according
        # to self.multi_mode
        self.segs = {
            "kidney_cortex" : {
                "dir" : "t1_clean",
                "glob" : "seg_kidney_*_cortex_t1*.nii.gz"
            },
            "kidney_cortex_l" : {
                "dir" : "t1_clean",
                "glob" : "seg_kidney_*_cortex_l_*.nii.gz"
            },
            "kidney_cortex_r" : {
                "dir" : "t1_clean",
                "glob" : "seg_kidney_*_cortex_r_*.nii.gz"
            },
            "kidney_medulla" : {
                "dir" : "t1_clean",
                "glob" : "seg_kidney_*_medulla_t1_*.nii.gz"
            },
            "kidney_medulla_l" : {
                "dir" : "t1_clean",
                "glob" : "seg_kidney_*_medulla_l_*.nii.gz"
            },
            "kidney_medulla_r" : {
                "dir" : "t1_clean",
                "glob" : "seg_kidney_*_medulla_r_*.nii.gz"
            },
            "tkv_l" : {
                "dir" : "t2w",
                "glob" : "seg_t2w_left_kidney.nii.gz",
            },
            "tkv_r" : {
                "dir" : "t2w",
                "glob" : "seg_t2w_right_kidney.nii.gz",
            }
        }

        # Mapping from name to information defining parameter maps. 
        # The glob matches one or more parameter map files. If multiple map/seg combinations are
        # found they are handled according to self.multi_mode
        # Limits are used to exclude out of range values from the statistics
        self.params = {
            "t2star_exp" : {
                "dir" : "t2star",
                "glob" : "t2star_2p_exp.nii.gz",
                "limits" : (2, 100),
            },
            "t2star_loglin" : {
                "dir" : "t2star",
                "glob" : "t2star_loglin.nii.gz",
                "limits" : (2, 100),
            },
            "r2star_exp" : {
                "dir" : "t2star",
                "glob" : "r2star_2p_exp.nii.gz",
                "limits" : (10, 500),
            },
            "r2star_loglin" : {
                "dir" : "t2star",
                "glob" : "r2star_loglin.nii.gz",
                "limits" : (10, 500),
            },
            "t1" : {
                "dir" : "align",
                "glob" : "t1_conf_*.nii.gz",
                "limits" : (1000, 2500),
            },
            "mtr" : {
                "dir" : "mtr",
                "glob" : "mtr.nii.gz",
            },
            "b0" : {
                "dir" : "b0",
                "glob" : "b0.nii.gz",
                "segs" : ["tkv_l", "tkv_r"],
            },
            "b1" : {
                "dir" : "b1",
                "glob" : "b1.nii.gz",
                "segs" : ["tkv_l", "tkv_r"],
            },
        }

        # Standard set of statistics to use in outpu CSV file
        self.stats = ["n", "iqn", "iqmean", "median", "iqstd"]

    def get_file(indir, path):
        LOG.debug(f" - Looking for file from {indir} matching path: {path}")

        glob_path = os.path.join(indir, path)
        files = sorted(list(glob.glob(glob_path)))
        if len(files) == 0:
            LOG.warn(f" - No matching files found")
            return []

        return files

    def process(self):
        stat_names, values = [], []
        for param, param_spec in self.params.items():
            param_start = len(values)
            for seg, seg_spec in self.segs.items():
                if "segs" in param_spec and seg not in param_spec["segs"]:
                    LOG.debug(f" - Skipping segmentation {seg} for param {param}")
                    continue
                stats_data, res_imgs, n_found, best_count = [], [], 0, 0

                LOG.info(f" - Generating stats for param {param}, segmentation {seg}")
                LOG.debug(param_spec)
                LOG.debug(seg_spec)
                for param_img in self.inimgs(param_spec["dir"], param_spec["glob"], is_depfile=True):
                    for seg_img in self.inimgs(seg_spec["dir"], seg_spec["glob"], is_depfile=True):
                        res_nii = self.resample(seg_img, param_img, is_roi=True)
                        res_data = res_nii.get_fdata()
                        orig_count = np.count_nonzero(seg_img.data)
                        res_count = np.count_nonzero(res_data)
                        n_found += 1 if res_count > 0 else 0
                        LOG.debug(f" - Param {param_img.fname}, Seg {seg_img.fname} count {res_count} orig {orig_count}")
                        if self.multi_mode == "best":
                            if res_count > best_count:
                                stats_data = [param_img.data[res_data > 0]]
                                res_imgs = [res_nii]
                        elif self.multi_mode == "combine":
                            if res_count > 0:
                                stats_data.append(param_img.data[res_data > 0])
                                res_imgs.append(res_nii)
                        else:
                            raise RuntimeError(f"Multi mode not recognized: {self.multi_mode}")

                if n_found == 0:
                    LOG.warn(" - No combination found with overlap")
                elif self.multi_mode == "best" and n_found != 1:
                    LOG.warn(f" - {n_found} combinations found with overlap - choosing best")
                elif self.multi_mode == "combine":
                    LOG.debug(f" - Combining data from {n_found} overlapping parameter/segmentation maps")

                #voxel_vol = 0
                for idx, res_img in enumerate(res_imgs):
                    res_path = self.outfile(f"{seg}_res_{param}_{idx+1}.nii.gz")
                    LOG.debug(f" - Saving resampled segmentation to {res_path}")
                    res_img.to_filename(res_path)
                    #voxel_vol = util.voxel_vol(res_img)
                if stats_data:
                    stats_data = np.concatenate(stats_data)
                param_stats = fproc.stats.run(stats_data, stats=self.stats, data_limits=param_spec.get("limits", (None, None)))
                #param_stats["vol"] = param_stats["n"] * voxel_vol
                #param_stats["iqvol"] = param_stats["iqn"] * voxel_vol
                data_colname = param + "_" + seg
                for stat, value in param_stats.items():
                    stat_names.append(stat + "_"+ data_colname)
                    values.append(value)

            stats_path = self.outfile("stats.csv")
            LOG.info(f" - Saving stats to {stats_path}")
            with open(stats_path, "w") as stats_file:
                for name, value in zip(stat_names, values):
                    stats_file.write(f"{name},{str(value)}\n")

class CMD(Module):
    def __init__(self):
        Module.__init__(self, "cmd")
        
    def process(self):
        stats_file = os.path.join(self.pipeline.options.output, "stats", "stats.csv")
        cmd_dict = {}
        with open(stats_file) as f:
            for line in f:
                try:
                    key, value = line.split(",")
                    value = float(value)
                    if "iqmean" not in key and "median" not in key:
                        # Only calculate CMD for IQ mean and median
                        continue
                    for struc in ("cortex", "medulla"):
                        if struc in key:
                            generic_key = key.replace(struc, "cmd")
                            if generic_key not in cmd_dict:
                                cmd_dict[generic_key] = {}
                            cmd_dict[generic_key][struc] = value
                except Exception as exc:
                    LOG.warn(f"Error parsing stats file {stats_file} line {line}: {exc}")

        stats_path = self.outfile("cmd.csv")
        LOG.info(f" - Saving CMD stats to {stats_path}")
        with open(stats_path, "w") as stats_file:
            for key, values in cmd_dict.items():
                if "cortex" not in values or "medulla" not in values:
                    LOG.warn(f"Failed to find both cortex and medulla data for key: {key}")
                    continue
                cmd = values["medulla"] - values["cortex"]
                stats_file.write(f"{key},{str(cmd)}\n")

class Volumes(Module):
    def __init__(self):
        Module.__init__(self, "volumes")

    def _mask_vol(self, fname):
        if os.path.exists(fname):
            img = ImageFile(fname)
            return img.voxel_volume * np.count_nonzero(img.data)
        else:
            return 0
        
    def process(self):
        t1_all_segs = self.inimgs("t1_clean", "seg_kidney*_all_t1*.nii.gz", is_depfile=True)
        if not t1_all_segs:
            self.no_data(" - No T1 segmentations found for volume comparison")
            
        t2w_all_segs = self.inimgs("resample", "seg_t2w*_mask_*.nii.gz", is_depfile=True)
        if not t2w_all_segs:
            self.no_data(" - No T2w masks found to for volume comparison")

        tkv_vol_left, tkv_vol_right = 0, 0
        for t2w_seg in t2w_all_segs:
            LOG.info(f"Adding TKV volumes from {t2w_seg.fname}")
            tkv_left_fname = t2w_seg.fpath.replace("_mask_", "_left_kidney_")
            tkv_right_fname = t2w_seg.fpath.replace("_mask_", "_right_kidney_")
            tkv_vol_left += self._mask_vol(tkv_left_fname)
            tkv_vol_right += self._mask_vol(tkv_right_fname)

        t1_vol_left, t1_vol_right = 0, 0
        for t1_seg in t1_all_segs:
            LOG.info(f"Adding T1 volumes from {t1_seg.fname}")
            t1_left_fname = t1_seg.fpath.replace("_all_", "_all_l_")
            t1_right_fname = t1_seg.fpath.replace("_all_", "_all_r_")
            t1_vol_left += self._mask_vol(t1_left_fname)
            t1_vol_right += self._mask_vol(t1_right_fname)

        LOG.info(f" - Saving T1/T2w volume comparison to t1_t2w_volumes.csv")
        with open(self.outfile("t1_t2w_volumes.csv"), "w") as f:
            f.write(f"tkv_vol_left, {tkv_vol_left}\n")
            f.write(f"tkv_vol_right, {tkv_vol_right}\n")
            f.write(f"t1_vol_left, {t1_vol_left}\n")
            f.write(f"t1_vol_right, {t1_vol_right}\n")
            f.write(f"vol_fraction_left, {t1_vol_left/tkv_vol_left}\n")
            f.write(f"vol_fraction_right, {t1_vol_right/tkv_vol_right}\n")

class ShapeMetrics(Module):
    def __init__(self):
        Module.__init__(self, "shape_metrics")

    def process(self):
        METRICS_MAPPING = {
            'Surface area': "surf_area",
            'Volume': "vol",
            'Bounding box volume': "vol_bb",
            'Convex hull volume': "vol_ch",
            'Volume of holes': "vol_holes",
            'Extent': "extent",
            'Solidity': "solidity",
            'Compactness': "compactness",
            'Long axis length': "long_axis",
            'Short axis length': "short_axis",
            'Equivalent diameter': "equiv_diam",
            'Longest caliper diameter': "longest_diam",
            'Maximum depth': "max_depth",
            'Primary moment of inertia': "mi1",
            'Second moment of inertia': "mi2",
            'Third moment of inertia': "mi3",
            'Mean moment of inertia': "mi_mean",
            'Fractional anisotropy of inertia': "fa",
            'QC - Volume check': "volcheck",
        }

        t2w_all_segs = self.inimgs("t2w", "seg_t2w_mask.nii.gz", is_depfile=True)
        if not t2w_all_segs:
            self.no_data(" - No T2w masks found to for shape metrics")

        LOG.info(f" - Saving T2w shape metrics to t2w_shape_metrics.csv")
        with open(self.outfile("t2w_shape_metrics.csv"), "w") as f:
            for side in ("left", "right"):
                img = self.inimg("t2w", f"seg_t2w_{side}_kidney.nii.gz", is_depfile=True)
                LOG.info(f" - Calculating T2w shape metrics from {img.fname}")
                try:
                    vol_metrics = _volume_features(img.data, affine=img.affine)
                except Exception as exc:
                    LOG.warn(f"Failed to calculate volume features: {exc}")
                for metric, value in vol_metrics.items():
                    value, units = value
                    col_name = f"tkv_{side}_" + METRICS_MAPPING[metric]
                    f.write(f"{col_name},{value}\n")

MODULES = [
    B0(),
    B1(),
    MTR(),
    T1(),
    T1w(),
    T2star(),
    T2w(),
    Alignment(),
    Resample(),
    T1Clean(),
    SegStats(),
    CMD(),
    Volumes(),
    ShapeMetrics(),
]

class RenalPreprocArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "renal_preproc", __version__)
        self.add_argument("--t2w-fixed-masks", help="Directory containing fixed T2w masks")
        self.add_argument("--t2star-method", help="Method to use when doing T2* processing", choices=["loglin", "2p_exp", "all"], default="all")
        self.add_argument("--t2star-resample", help="Planar resolution to resample T2* to (mm)", type=float, default=0)
        self.add_argument("--t2w-model", "--segmentation-weights", help="Filename or URL for T2w segmentation CNN weights", default="whole_kidney_cnn.model")
        self.add_argument("--t1-model", help="Filename or URL for T1 segmentation model weights", default="/spmstore/project/RenalMRI/trained_models/kidney_t1_molli_min_max.pt")
        self.add_argument("--t1-no-realign", help="Don't try to realign T1 segmentations to T2*", action="store_true", default=False)

class RenalPreproc(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "renal preproc", __version__, RenalPreprocArgumentParser(), MODULES)

if __name__ == "__main__":
    renal_preproc = RenalPreproc()
    renal_preproc.run()
