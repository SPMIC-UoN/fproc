"""
FPROC: Segmentations of various body parts
"""
import logging
import validators
import wget
import glob

import numpy as np
import skimage
from scipy.ndimage import binary_fill_holes, binary_erosion, generate_binary_structure, binary_dilation
import radiomics

from fsort import ImageFile
from fproc.module import Module

LOG = logging.getLogger(__name__)

class KneeToNeckDixon(Module):
    def __init__(self, name="seg_knee_to_neck_dixon", **kwargs):
        self._dixon_dir = kwargs.get("dixon_dir", "dixon")
        deps = [self._dixon_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        LOG.info(f" - Doing DIXON knee-to-nexck segmentation for subject")
        src = self.kwargs.get("dixon_src", self.INPUT)
        knee_to_neck_model = self.pipeline.options.knee_to_neck_dixon_model
        fat = self.inimg(self._dixon_dir, "fat.nii.gz", src=src)
        t2star = self.inimg(self._dixon_dir, "t2star.nii.gz", src=src)
        water = self.inimg(self._dixon_dir, "water.nii.gz", src=src)

        ip_data = water.data + fat.data
        op_data = water.data - fat.data
        mask_data = t2star.data > 0
        water.save_derived(ip_data, self.outfile("ip.nii.gz"))
        water.save_derived(op_data, self.outfile("op.nii.gz"))
        water.save_derived(mask_data, self.outfile("mask.nii.gz"))
        self.runcmd([
            'infer_knee_to_neck_dixon',
            '--output_folder', self.outdir,
            '--reference_header_nifti', water.fpath,
            '--save_what', 'prob'
            '--threshold_method', 'fg',
            '--fat', fat.fpath,
            '--water', water.fpath,
            '--inphase', self.outfile("ip.nii.gz"),
            '--outphase', self.outfile("op.nii.gz"),
            '--mask', self.outfile("mask.nii.gz"),
            '--restore_string', knee_to_neck_model,
        ], logfile=f'seg.log')

class LiverDixon(Module):
    def __init__(self, name="seg_liver_dixon", **kwargs):
        self._dixon_dir = kwargs.get("dixon_dir", "dixon")
        deps = [self._dixon_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        src = self.kwargs.get("dixon_src", self.INPUT)
        fat = self.inimg(self._dixon_dir, "fat.nii.gz", src=src)
        t2star = self.inimg(self._dixon_dir, "t2star.nii.gz", src=src)
        water = self.inimg(self._dixon_dir, "water.nii.gz", src=src)
        self.run_nnunetv2("14", [fat, t2star, water], "liver", "DIXON", water, "water")


class SpleenDixon(Module):
    def __init__(self, name="seg_spleen_dixon", **kwargs):
        self._dixon_dir = kwargs.get("dixon_dir", "dixon")
        deps = [self._dixon_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        src = self.kwargs.get("dixon_src", self.INPUT)
        fat = self.inimg(self._dixon_dir, "fat.nii.gz", src=src)
        t2star = self.inimg(self._dixon_dir, "t2star.nii.gz", src=src)
        water = self.inimg(self._dixon_dir, "water.nii.gz", src=src)
        self.run_nnunetv2("102", [fat, t2star, water], "spleen", "DIXON", water, "water")

class PancreasEthrive(Module):
    def __init__(self, name="seg_pancreas_ethrive", **kwargs):
        self._ethrive_dir = kwargs.get("ethrive_dir", "ethrive")
        deps = [self._ethrive_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        ethrive_glob = self.kwargs.get("ethrive_glob", "ethrive.nii.gz")
        ethrive = self.inimg(self._ethrive_dir, ethrive_glob)
        self.run_nnunetv2("234", [ethrive], "pancreas", "eTHRIVE", ethrive, "ethrive")

class KidneyCystT2w(Module):
    def __init__(self, name="seg_kidney_cyst_t2w", **kwargs):
        self._t2w_dir = kwargs.get("t2w_dir", "t2w")
        deps = [self._t2w_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        t2w_glob = self.kwargs.get("t2w_glob", "t2w.nii.gz")
        t2w_src = self.kwargs.get("t2w_src", self.INPUT)
        t2w_map = self.single_inimg(self._t2w_dir, t2w_glob, src=t2w_src)
        self.run_nnunetv2("244", [t2w_map], "kidney_cyst", "T2w")

        # Remove normal kidney from mask
        seg = self.inimg(self.name, "kidney_cyst.nii.gz", src=self.OUTPUT)
        cyst_mask = np.copy(seg.data)
        cyst_mask[cyst_mask == 2] = 0
        cyst_mask = (cyst_mask > 0).astype(np.int8)
        seg.save_derived(cyst_mask, self.outfile("kidney_cyst_mask.nii.gz"))

        mask = self.inimg(self.name, "kidney_cyst_mask.nii.gz", src=self.OUTPUT)
        self.lightbox(t2w_map, mask, name="kidney_cyst_t2w_lightbox", tight=True)
        
        # Count number of cysts and volume
        total_volume = np.count_nonzero(mask.data) * mask.voxel_volume
        labelled = skimage.measure.label(mask.data)
        props = skimage.measure.regionprops(labelled)
        num_cysts = len(props)

        with open(self.outfile("kidney_cyst.csv"), "w") as f:
            f.write(f"vol_kidney_cyst,{total_volume}\n")
            f.write(f"count_kidney_cysts,{num_cysts}\n")


class KidneyT1(Module):
    def __init__(self, name="seg_kidney_t1", map_dir="t1_kidney", map_glob="t1_map*.nii.gz", **kwargs):
        self._dir = map_dir
        self._glob = map_glob
        deps = [self._dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        t1_maps = self.inimgs(self._dir, self._glob, src=self.kwargs.get("map_src", self.OUTPUT))
        if not t1_maps:
            self.no_data(f"No T1 maps found to segment in {self._dir}/{self._glob}")
        t1_limits = self.kwargs.get("t1_limits", None)

        single_map = len(t1_maps) == 1
        for t1_map in t1_maps:
            LOG.info(f" - Segmenting KIDNEY using T1 data: {t1_map.fname}")
            t1_data = t1_map.data
            if t1_limits is not None:
                if len(t1_limits) != 2 or len(t1_limits[0]) != 2 or len(t1_limits[1]) != 2:
                    self.bad_data("Invalid T1 limits format - should be sequence of two tuples ((min, replace), (max, replace)) - ignoring")
                LOG.info(f" - Replacing T1 < {t1_limits[0][0]} with {t1_limits[0][1]} and T1 > {t1_limits[1][0]} with {t1_limits[1][1]}") 
                t1_data[t1_data < t1_limits[0][0]] = t1_limits[0][1]
                t1_data[t1_data > t1_limits[1][0]] = t1_limits[1][1]

            if single_map:
                out_prefix = "kidney"
                t1_map = t1_map.save_derived(t1_data, self.outfile("t1_map.nii.gz"))
            else:
                out_prefix = f'kidney_{t1_map.fname_noext}'
                t1_map = t1_map.save_derived(t1_data, self.outfile(t1_map.fname))
 
            self.runcmd([
                'kidney_t1_seg',
                '--input', t1_map.dirname,
                '--subjid', '',
                '--display-id', self.pipeline.options.subjid,
                '--t1', t1_map.fname,
                '--model', self.pipeline.options.kidney_t1_model,
                '--noclean',
                '--output', self.outdir,
                '--outprefix', out_prefix],
                logfile=f'seg.log'
            )

            seg = self.inimg(self.name, f"{out_prefix}_all_t1.nii.gz", src=self.OUTPUT)
            self.lightbox(t1_map, seg, name=f"{out_prefix}_t1_lightbox", tight=True)

class KidneyT2w(Module):
    def __init__(self, name="seg_kidney_t2w", **kwargs):
        self._t2w_dir = kwargs.get("t2w_srcdir", "t2w")
        deps = [self._t2w_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        t2w_map = self.inimg(self._t2w_dir, "t2w.nii.gz")
        LOG.info(f" - Segmenting KIDNEY using T2w data: {t2w_map.fname}")

        model_weights = self.kwargs.get("model", self.pipeline.options.kidney_t2w_model)
        if validators.url(model_weights):
            # Download weights from supplied URL
            LOG.info(f" - Downloading model weights from {model_weights}")
            wget.download(model_weights, "model.h5")
            model_weights = "model.h5"
        else:
            LOG.info(f" - Using model weights from {model_weights}")

        # The segmentor needs both the image array and affine so the size of each voxel is known. post_process=True removes all but
        # the largest two areas in the mask e.g. removes small areas of incorrectly categorised tissue. This can cause issues if the
        # subject has more or less than two kidneys though.
        from ukat.segmentation import whole_kidney
        segmentation = whole_kidney.Segmentation(t2w_map.data, t2w_map.affine, post_process=True, binary=True, weights=model_weights)
        segmentation.to_nifti(output_directory=self.outdir, base_file_name=f"kidney", maps=['mask', 'left', 'right', 'individual'])

        LOG.info(f" - Generating overlay image for T2w segmentation using {t2w_map.fname}")
        mask_img = ImageFile(self.outfile("kidney_mask.nii.gz"), warn_json=False)
        self.lightbox(t2w_map, mask_img, "kidney_t2w_lightbox")

        vols_fname = self.kwargs.get("vols_fname", "tkv.csv")
        if vols_fname:
            LOG.info(f" - Saving volumes to {vols_fname}")
            with open(self.outfile(vols_fname), "w") as f:
                for roi in ["left", "right", "mask"]:
                    img = self.single_inimg(self.outdir, f"*{roi}*.nii.gz")
                    name = roi if roi != "mask" else "total"
                    voxel_vol = img.voxel_volume
                    vol = voxel_vol * np.count_nonzero(img.data)
                    f.write(f"kv_{name},{vol}\n")


class KidneyT2wRenalSegmentor(Module):
    def __init__(self, name="seg_kidney_t2w", **kwargs):
        self._t2w_dir = kwargs.get("t2w_srcdir", "t2w")
        deps = [self._t2w_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        t2w_map = self.inimg(self._t2w_dir, "t2w.nii.gz")
        LOG.info(f" - Segmenting KIDNEY using renal segmentor - T2w data: {t2w_map.fname}")

        from segment import Tkv
        segmentation = Tkv(t2w_map.fpath)
        mask = segmentation.get_mask()
        mask_img = t2w_map.save_derived(mask.astype(np.int8), self.outfile("kidney_mask.nii.gz"))
        mask_left = self.split_lr(mask, affine=t2w_map.affine, side='left')
        mask_right = self.split_lr(mask, affine=t2w_map.affine, side='right')
        t2w_map.save_derived(mask_left.astype(np.int8), self.outfile("kidney_left.nii.gz"))
        t2w_map.save_derived(mask_right.astype(np.int8), self.outfile("kidney_right.nii.gz"))

        LOG.info(f" - Generating overlay image for T2w segmentation using {t2w_map.fname}")
        self.lightbox(t2w_map, mask_img, "kidney_t2w_lightbox")

        vols_fname = self.kwargs.get("vols_fname", "tkv.csv")
        if vols_fname:
            LOG.info(f" - Saving volumes to {vols_fname}")
            with open(self.outfile(vols_fname), "w") as f:
                for roi in ["left", "right", "mask"]:
                    img = self.single_inimg(self.outdir, f"*{roi}*.nii.gz")
                    name = roi if roi != "mask" else "total"
                    voxel_vol = img.voxel_volume
                    vol = voxel_vol * np.count_nonzero(img.data)
                    f.write(f"kv_{name},{vol}\n")


class KidneyT1SE(Module):
    def __init__(self, name="seg_kidney_t1_se", **kwargs):
        self._t1_se_dir = kwargs.get("t1_se_dir", "t1_se")
        self._ref_dir = kwargs.get("t1_ref_dir", "../fsort/t1_se_raw")
        deps = [self._t1_se_dir, self._ref_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        t1_se_glob = self.kwargs.get("t1_se_glob", "t1_se*.nii.gz")
        t1_ref_glob = self.kwargs.get("t1_ref_glob", "*.nii.gz")

        t1_se_data = self.inimgs(self._t1_se_dir, t1_se_glob, src=self.OUTPUT)
        if not t1_se_data:
            self.no_data(f"No T1 SE data found in {self._t1_se_dir}/{t1_se_glob}")

        t1_ref_data = self.inimgs(self._ref_dir, t1_ref_glob)
        if not t1_ref_data:
            self.no_data(f"No T1 SE reference data found in {self._ref_dir}/{t1_ref_glob}")
        vendor = t1_ref_data[0].vendor

        single_map = len(t1_se_data) == 1
        for t1_se_img in t1_se_data:
            LOG.info(f" - Segmenting KIDNEY using T1 SE data: {t1_se_img.fname} from vendor {vendor}")
            if single_map:
                out_prefix = "kidney"
            else:
                out_prefix = f'kidney_{t1_se_img.fname_noext}'
            self.runcmd([
                'seg_kidney_t1_se',
                '--data', t1_se_img.fpath,
                '--model', "minmax",
                '--output', self.outdir,
                '--output-prefix', out_prefix,
                '--vendor', vendor],
                logfile=f'seg.log'
            )

            #medulla = ImageFile(self.outfile(f"{out_prefix}_medulla.nii.gz"))
            #self.lightbox(t1_se_img, medulla, name=f"{out_prefix}_t1_lightbox", tight=True)

class SatDixon(Module):
    def __init__(self, name="seg_sat_dixon", **kwargs):
        self._dixon_dir = kwargs.get("dixon_dir", "dixon")
        deps = [self._dixon_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        src = self.kwargs.get("dixon_src", self.INPUT)
        fat = self.inimg(self._dixon_dir, "fat.nii.gz", src=src)
        water = self.inimg(self._dixon_dir, "water.nii.gz", src=src)
        self.run_nnunetv2("141", [fat], "sat", "DIXON", water, "water")

class BodyDixon(Module):
    def __init__(self, name="seg_body_dixon", **kwargs):
        self._dixon_dir = kwargs.get("dixon_dir", "dixon")
        deps = [self._dixon_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        """
        Estimate body mask from dixon water map
        """
        src = self.kwargs.get("dixon_src", self.INPUT)
        water_thresh = self.kwargs.get("water_thresh", 20)
        water = self.inimg(self._dixon_dir, "water.nii.gz", src=src)
        LOG.info(f" - Segmenting body by thresholding water map {water.fname} at level {water_thresh}")

        # Work slicewise and try to segment the body by thresholding, filling holes and
        # selecting the largest contiguous region (blob)
        water_data = water.data.squeeze()
        body_mask = []
        for sl in range(water.shape[2]):
            water_data_slice = water_data[..., sl]
            water_data_nonzero = water_data_slice[water_data_slice > 0]
            if water_data_nonzero.size == 0:
                body_mask.append(np.zeros(water_data_slice.shape, dtype=np.int8))
                continue
            thresh = np.percentile(water_data_nonzero, water_thresh)
            mask_slice = (water_data_slice > thresh).astype(np.int8).squeeze()
            mask_slice_filled = binary_fill_holes(mask_slice)
            largest_blob = self.blobs_by_size(mask_slice_filled, min_size=10)[0]
            body_mask.append(largest_blob)
        body_mask = np.stack(body_mask, axis=-1)

        water.save_derived(body_mask, self.outfile("body.nii.gz"))
        seg = self.inimg(self.name, "body.nii.gz", src=self.OUTPUT)
        self.lightbox(water, seg, name="body_water_lightbox", tight=True)

class VatDixon(Module):
    def __init__(self, name="seg_vat_dixon", **kwargs):
        self._ff_dir = kwargs.get("ff_dir", "fat_fraction")
        self._body_dir = kwargs.get("body_dir", "seg_body_dixon")
        self._sat_dir = kwargs.get("sat_dir", "seg_sat_dixon")
        self._dixon_dir = kwargs.get("dixon_dir", "dixon")
        deps = [self._ff_dir, self._body_dir, self._sat_dir, self._dixon_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        """
        Get VAT from all fat - (SAT + organ masks)
        """
        ff_src = self.kwargs.get("ff_src", self.OUTPUT)
        ff_thresh = self.kwargs.get("ff_thresh", 85)
        ff_glob = self.kwargs.get("ff_glob", "fat_fraction.nii.gz")
        ff = self.inimg(self._ff_dir, ff_glob, src=ff_src)
        body = self.inimg(self._body_dir, "body.nii.gz", src=self.OUTPUT)
        sat_glob = self.kwargs.get("sat_glob", "sat.nii.gz")
        sat = self.inimg(self._sat_dir, sat_glob, src=self.OUTPUT)
        vat_data = (ff.data > ff_thresh).astype(np.int8)
        while vat_data.ndim > 3:
            vat_data = vat_data.squeeze(-1)
        vat_data[body.data == 0] = 0
        ff.save_derived(vat_data, self.outfile("fat.nii.gz"))

        vat_data[sat.data > 0] = 0
        organs = self.kwargs.get("organs", {})
        for organ_dir, fname in organs.items():
            organ_seg = self.inimg(organ_dir, fname, src=self.OUTPUT, check=False)
            if organ_seg is None:
                msg = f"Could not find segmentation: {organ_dir}/{fname}"
                if self.kwargs.get("fail_on_missing", True):
                    self.bad_data(msg)
                else:
                    LOG.warn(msg)
            else:
                LOG.info(f" - Removing organ {organ_dir}/{fname} from VAT")
                res_data = self.resample(organ_seg, ff, is_roi=True, allow_rotated=True).get_fdata().astype(np.int8)
                vat_data[res_data > 0] = 0
                ff.save_derived(res_data, self.outfile(fname.replace(".nii.gz", "_res.nii.gz")))

        if self.kwargs.get("prune_slices", True):
           # FIXME temporary remove top/bottom slices to avoid problem with SAT segmentor
            vat_data[..., 0] = 0
            vat_data[..., -1] = 0

        ff.save_derived(vat_data, self.outfile("vat.nii.gz"))
        seg = self.inimg(self.name, "vat.nii.gz", src=self.OUTPUT)

        dixon_src = self.kwargs.get("dixon_src", self.INPUT)
        water = self.inimg(self._dixon_dir, "water.nii.gz", src=dixon_src, check=False)
        if water is not None:
            self.lightbox(water, seg, name="vat_water_lightbox", tight=True)
        else:
            LOG.warn(" - No water map found to overlay with VAT")

class LegDixon(Module):
    def __init__(self, name="seg_leg_dixon", **kwargs):
        self._dixon_dir = kwargs.get("dixon_dir", "dixon")
        deps = [self._dixon_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        src = self.kwargs.get("dixon_src", self.INPUT)
        water = self.inimg(self._dixon_dir, "water.nii.gz", src=src)
        fat = self.inimg(self._dixon_dir, "fat.nii.gz", src=src)

        LOG.info(f" - Segmenting LEG using water: {water.fpath}, fat: {fat.fpath}")
        self.runcmd([
            'leg_dixon_seg',
            '--water', water.fpath,
            '--fat', fat.fpath,
            '--model', self.pipeline.options.leg_dixon_model,
            '--output', self.outfile("leg.nii.gz")
        ], logfile=f'seg.log')

        seg = self.inimg(self.name, "leg.nii.gz", src=self.OUTPUT)
        self.lightbox(water, seg, name="leg_water_lightbox", tight=True)


class OrganDixon(Module):
    def __init__(self, organ, **kwargs):
        self._dixon_dir = kwargs.get("dixon_dir", "dixon")
        self.organ = organ
        name = kwargs.pop("name", f"seg_{organ}_dixon")
        deps = [self._dixon_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        src = self.kwargs.get("dixon_src", self.INPUT)
        inputs = self.kwargs.get("inputs", ["fat", "fat_fraction", "t2star", "water"])
        input_imgs = []
        for input in inputs:
            input_imgs.append(self.inimg(self._dixon_dir, f"{input}.nii.gz", src=src))
        model_id = self.kwargs.get("model_id", "422")
        water = self.inimg(self._dixon_dir, "water.nii.gz", src=src)
        self.run_nnunetv2(model_id, input_imgs, self.organ, "DIXON", water, "water")

        if self.kwargs.get("splitlr", self.organ == "kidney"):
            organ_img = self.inimg(self.name, f"{self.organ}.nii.gz", src=self.OUTPUT)
            left = self.split_lr(organ_img.data, organ_img.affine, "l")
            right = self.split_lr(organ_img.data, organ_img.affine, "r")
            organ_img.save_derived(left, self.outfile("kidney_left.nii.gz"))
            organ_img.save_derived(right, self.outfile("kidney_right.nii.gz"))

class KidneyDixon(OrganDixon):
    def __init__(self, **kwargs):
        OrganDixon.__init__(self, "kidney", **kwargs)

class KidneyCortexMedullaT2w(Module):
    """
    Cortex/medulla masking using T2w whole kidney segmentation
    """
    def __init__(self, name="seg_kidney_cortex_medulla_t2w", **kwargs):
        self._t2w_seg_dir = kwargs.get("t2w_seg_dir", "seg_kidney_t2w")
        self._t2star_dir = kwargs.get("t2star_dir", "t2star")
        deps = [self._t2w_seg_dir, self._t2star_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        import fsl.wrappers as fsl

        t2w_seg_fname = self.kwargs.get("t2w_seg_fname", "kidney_mask.nii.gz")
        t2w_seg = self.single_inimg(self._t2w_seg_dir, t2w_seg_fname, src=self.OUTPUT)
        if t2w_seg is None:
            self.no_data(f"No T2w kidney segmentation found in {self._t2w_seg_dir}/{t2w_seg_fname}")
        LOG.info(f" - T2w kidney segmentation shape: {t2w_seg.shape}")

        last_echo_glob = self.kwargs.get("t2star_last_echo_glob", "last_echo.nii.gz")
        t2star_last_echo = self.single_inimg(self._t2star_dir, last_echo_glob, src=self.OUTPUT)
        if t2star_last_echo is None:
            self.no_data(f"No T2* last echo found in {self._t2star_dir}/{last_echo_glob}")
        LOG.info(f" - last echo shape: {t2star_last_echo.shape}")

        flirt_result = fsl.flirt(t2w_seg.nii, t2star_last_echo.nii, out=fsl.LOAD, omat=fsl.LOAD, usesqform=True, applyxfm=True)
        masktot2star = flirt_result["out"].get_fdata().reshape(t2star_last_echo.shape)
        LOG.info(f" - masktot2star shape: {masktot2star.shape}")
        t2star_last_echo.save_derived(masktot2star, self.outfile("masktot2star.nii.gz"))
        final_echo_masked = np.copy(t2star_last_echo.data)
        final_echo_masked[masktot2star == 0] = 0
        final_echo_masked_mean = np.mean(final_echo_masked[masktot2star > 0])

        final_echo_thr = np.copy(final_echo_masked)
        final_echo_thr[final_echo_thr < final_echo_masked_mean] = 0
        final_echo_thr_20 = np.copy(final_echo_masked)
        final_echo_thr_20[final_echo_thr_20 < final_echo_masked_mean * 1.2] = 0

        cortex_mask = (final_echo_thr > 0).astype(np.int32)
        cortex_mask_stats = (final_echo_thr_20 > 0).astype(np.int32)

        struct = generate_binary_structure(3, 1)
        struct[..., 0] = 0
        struct[..., -1] = 0
        masktot2star_bin = (masktot2star > 0).astype(np.int32)
        t2star_last_echo.save_derived(masktot2star_bin, self.outfile("masktot2star_bin.nii.gz"))

        masktot2star_ero = binary_erosion(masktot2star_bin, iterations=2, structure=struct).reshape(t2star_last_echo.shape)
        LOG.info(f" - masktot2star_ero shape: {masktot2star_ero.shape}, {np.count_nonzero(masktot2star_ero)}")
        t2star_last_echo.save_derived(masktot2star_ero, self.outfile("masktot2star_ero.nii.gz"))

        remainder_mask = masktot2star_ero - cortex_mask
        LOG.info(f" - remainder mask shape: {remainder_mask.shape}")
        t2star_last_echo.save_derived(remainder_mask, self.outfile("remainder_mask.nii.gz"))

        final_echo_rest = np.copy(final_echo_masked)
        final_echo_rest[final_echo_rest < final_echo_masked_mean * 0.5] = 0

        rest = (final_echo_rest > 0).astype(np.int32)
        LOG.info(f" - rest shape: {rest.shape}")
        t2star_last_echo.save_derived(rest, self.outfile("rest.nii.gz"))

        medulla_mask = np.copy(remainder_mask)
        medulla_mask[rest == 0] = 0

        medulla_mask_bin = (medulla_mask > 0).astype(np.int32)

        t2star_last_echo.save_derived(cortex_mask, self.outfile("cortex_mask.nii.gz"))
        t2star_last_echo.save_derived(cortex_mask_stats, self.outfile("cortex_mask_stats.nii.gz"))
        t2star_last_echo.save_derived(medulla_mask_bin, self.outfile("medulla_mask.nii.gz"))


class KidneyPelvisT2w(Module):
    def __init__(self, name="seg_kidney_pelvis_t2w", **kwargs):
        self._t2w_seg_dir = kwargs.get("t2w_seg_dir", "seg_kidney_t2w")
        self._t1_seg_dir = kwargs.get("t1_seg_dir", "seg_kidney_t1")
        self._t2w_map_dir = kwargs.get("t2w_map_dir", "t2w")
        deps = [self._t2w_seg_dir, self._t1_seg_dir, self._t2w_map_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        t2w_seg_fname = self.kwargs.get("t2w_seg_fname", "kidney_mask.nii.gz")
        t2w_seg = self.single_inimg(self._t2w_seg_dir, t2w_seg_fname, src=self.OUTPUT)
        if t2w_seg is None:
            self.no_data(f"No T2w kidney segmentation found in {self._t2w_seg_dir}/{t2w_seg_fname}")
        LOG.info(f" - T2w kidney segmentation shape: {t2w_seg.shape}")

        cortex_glob = self.kwargs.get("t1_seg_cortex", "*cortex*.nii.gz")
        medulla_glob = self.kwargs.get("t1_seg_medulla", "*medulla*.nii.gz")
        cortex = self.inimgs(self._t1_seg_dir, cortex_glob, src=self.OUTPUT)
        medulla = self.inimgs(self._t1_seg_dir, medulla_glob, src=self.OUTPUT)
        if not cortex or not medulla:
            self.no_data(f"No T1 kidney segmentation found in {self._t1_seg_dir}")
        cortex_res = sum([self.resample(c, t2w_seg, is_roi=True, allow_rotated=True).get_fdata() for c in cortex])
        medulla_res = sum([self.resample(m, t2w_seg, is_roi=True, allow_rotated=True).get_fdata() for m in medulla])
        t2w_seg.save_derived(cortex_res.astype(np.int32), self.outfile("cortex_res.nii.gz"))
        t2w_seg.save_derived(medulla_res.astype(np.int32), self.outfile("medulla_res.nii.gz"))
        t1_kidney = (cortex_res + medulla_res > 0).astype(np.int32)
        t2w_seg.save_derived(t1_kidney, self.outfile("t1_kidney.nii.gz"))

        struct = generate_binary_structure(3, 1)
        struct[..., 0] = 0
        struct[..., -1] = 0
        kidney_fill = binary_fill_holes(t2w_seg.data)
        t2w_seg.save_derived(kidney_fill, self.outfile("kidney_fill.nii.gz"))
        kidney_fill_ero = binary_erosion(kidney_fill, iterations=1, structure=struct)
        t2w_seg.save_derived(kidney_fill_ero.astype(np.int32), self.outfile("kidney_fill_ero.nii.gz"))

        pelvis_data = (kidney_fill_ero - t1_kidney > 0).astype(np.int32)
        left_data = self.split_lr(pelvis_data, t2w_seg.affine, "left")
        right_data = self.split_lr(pelvis_data, t2w_seg.affine, "right")
        left_data = self.blobs_by_size(left_data, min_size=10)[0]
        right_data = self.blobs_by_size(right_data, min_size=10)[0]
        t2w_seg.save_derived(left_data.astype(np.int8), self.outfile("kidney_pelvis_left.nii.gz"))
        t2w_seg.save_derived(right_data.astype(np.int8), self.outfile("kidney_pelvis_right.nii.gz"))
        pelvis = t2w_seg.save_derived(pelvis_data, self.outfile("kidney_pelvis.nii.gz"))

        t2w_map_glob = self.kwargs.get("t2w_map_glob", "t2w.nii.gz")
        t2w_map = self.single_inimg(self._t2w_map_dir, t2w_map_glob)
        if t2w_map is not None:
            t2w_map.reorient2std()
            self.lightbox(t2w_map, pelvis, name="kidney_pelvis_lightbox", tight=True) 
        else:
            LOG.warn("No T2w map found - will not create lightbox image")


class KidneyPelvisTrace(Module):
    def __init__(self, name="seg_kidney_pelvis_trace", **kwargs):
        self._paren_dir = kwargs.get("paren_dir", "seg_kidney_t2w")
        self._whole_dir = kwargs.get("whole_dir", "traceseg")
        self._cyst_dir = kwargs.get("cyst_dir", "seg_kidney_cyst_t2w_trace")
        self._t2w_map_dir = kwargs.get("t2w_map_dir", "t2w")
        deps = [self._paren_dir, self._whole_dir, self._cyst_dir, self._t2w_map_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        paren_glob = self.kwargs.get("paren_glob", "kidney_mask.nii.gz")
        paren = self.single_inimg(self._paren_dir, paren_glob, src=self.OUTPUT)
        if paren is None:
            self.no_data("No kidney parenchyma segmentation found to define pelvis region")
        paren.reorient2std()
        paren.save_derived(paren.data, self.outfile("kidney_paren.nii.gz"))

        whole_glob = self.kwargs.get("whole_glob", "trace_kidney_all.nii.gz")
        whole = self.single_inimg(self._whole_dir, whole_glob, src=self.OUTPUT)
        if whole is None:
            self.no_data("No whole kidney segmentation found to define pelvis region")
        whole.reorient2std()

        cyst_glob = self.kwargs.get("cyst_glob", "kidney_cyst_fixed.nii.gz")
        cyst = self.single_inimg(self._cyst_dir, cyst_glob, src=self.OUTPUT)
        if cyst is None:
            self.no_data("No kidney cyst segmentation found to define pelvis region")
        cyst.reorient2std()

        whole_data = whole.data > 0
        whole_data = binary_fill_holes(whole_data)
        whole.save_derived(whole_data, self.outfile("whole_kidney_fill.nii.gz"))
        whole_data = binary_erosion(whole_data)
        whole.save_derived(whole_data, self.outfile("whole_kidney_ero.nii.gz"))
        cyst_paren_data = np.logical_or(cyst.data > 0, paren.data > 0)
        whole.save_derived(cyst_paren_data, self.outfile("kidney_cyst_paren.nii.gz"))
        pelvis_data = (whole_data.astype(np.int8) - cyst_paren_data.astype(np.int8)) > 0
        whole.save_derived(pelvis_data, self.outfile("pelvis_raw.nii.gz"))
        left_data = self.split_lr(pelvis_data, whole.affine, "left")
        right_data = self.split_lr(pelvis_data, whole.affine, "right")
        left_data = self.blobs_by_size(left_data, min_size=10)[0]
        right_data = self.blobs_by_size(right_data, min_size=10)[0]

        pelvis_data = left_data + right_data
        pelvis = whole.save_derived(pelvis_data, self.outfile("kidney_pelvis.nii.gz"))
        whole.save_derived(left_data.astype(np.int8), self.outfile("kidney_pelvis_left.nii.gz"))
        whole.save_derived(right_data.astype(np.int8), self.outfile("kidney_pelvis_right.nii.gz"))

        t2w_map_glob = self.kwargs.get("t2w_map_glob", "t2w.nii.gz")
        t2w_map = self.single_inimg(self._t2w_map_dir, t2w_map_glob)
        if t2w_map is not None:
            t2w_map.reorient2std()
            self.lightbox(t2w_map, pelvis, name="kidney_pelvis_lightbox", tight=True) 
        else:
            LOG.warn("No T2w map found - will not create lightbox image")


class KidneyFat(Module):
    def __init__(self, name="seg_kidney_fat", **kwargs):
        self._kidney_seg_dir = kwargs.get("kidney_seg_dir", "seg_kidney_dixon")
        self._ff_dir = kwargs.get("ff_dir", "fat_fraction")
        deps = [self._kidney_seg_dir, self._ff_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        kidney_seg_glob = self.kwargs.get("kidney_seg_glob", "kidney.nii.gz")
        ff_glob = self.kwargs.get("ff_glob", "fat_fraction.nii.gz")
        ff_thresh = self.kwargs.get("ff_thresh", 15)

        kidney = self.single_inimg(self._kidney_seg_dir, kidney_seg_glob, src=self.OUTPUT)
        if kidney is None:  
            self.no_data(f"No kidney segmentation found in {self._kidney_seg_dir}/{kidney_seg_glob}")
        kidney_filled = binary_fill_holes(kidney.data)

        ff = self.single_inimg(self._ff_dir, ff_glob, src=self.OUTPUT)
        if ff is None:
            self.no_data(f"No fat fraction data found in {self._ff_dir}/{ff_glob}")
        ff_data = self.resample(ff, kidney, is_roi=False, allow_rotated=True).get_fdata()
        fat_mask = ff_data > ff_thresh

        LOG.info(f" - Segmenting kidney fat / parenchyma using {kidney.fname}, {ff.fname} with threshold {ff_thresh}")
        kidney_parenchyma = np.copy(kidney_filled)
        kidney_parenchyma[fat_mask > 0] = 0

        fat_pelvis = np.copy(fat_mask)
        kidney_dil = binary_erosion(kidney_filled)
        fat_pelvis[kidney_dil == 0] = 0

        kidney.save_derived(ff_data, self.outfile("ff_res.nii.gz"))
        kidney.save_derived(fat_mask.astype(np.int32), self.outfile("fat_mask.nii.gz"))
        kidney.save_derived(kidney_filled.astype(np.int32), self.outfile("kidney_filled.nii.gz"))
        kidney.save_derived(kidney_dil.astype(np.int32), self.outfile("kidney_dil.nii.gz"))
        kidney.save_derived(kidney_parenchyma.astype(np.int32), self.outfile("kidney_parenchyma.nii.gz"))
        kidney.save_derived(fat_pelvis.astype(np.int32), self.outfile("fat_pelvis.nii.gz"))

        for side in ["left", "right"]:
            side_mask = self.split_lr(fat_pelvis, kidney.affine, side)
            kidney.save_derived(side_mask, self.outfile(f"fat_pelvis_{side}.nii.gz"))
            side_mask = self.split_lr(kidney_parenchyma, kidney.affine, side)
            kidney.save_derived(side_mask, self.outfile(f"kidney_parenchyma_{side}.nii.gz"))

class TotalSeg(Module):
    def __init__(self, name="totalseg", **kwargs):
        self._src_dir = kwargs.get("src_dir", "dixon")
        deps = [self._src_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        water_glob = self.kwargs.get("water_glob", "water.nii.gz")
        fat_glob = self.kwargs.get("fat_glob", "fat.nii.gz")
        water = self.single_inimg(self._src_dir, water_glob)
        if water is None:
            self.no_data(f"No input image found in {self._src_dir}/{water_glob}")
        LOG.info(f" - Running TotalSeg using water {water.fname}")
        crop = self.kwargs.get("crop", None)
        if crop:
            water = water.reorient2std()
            water_data = water.data
            zsize = water.shape[2]
            zmin = int(zsize * (1 - crop))
            water_data_cropped = water_data[:, :, zmin:]
            LOG.info(f" - Cropping input image retaining {crop} (discarding {zmin} slices)")
                    
            import nibabel as nib
            affine_cropped = np.copy(water.affine)
            affine_cropped[:, 3] += zmin * water.affine[:, 2]
            nii = nib.Nifti1Image(water_data_cropped, affine_cropped, water.nii.header)
            nii.to_filename(self.outfile("water_cropped.nii.gz"))

        self.runcmd([
                'TotalSegmentator',
                '-i', water.fpath,
                '-o', self.outdir,
                '--task', 'total_mr', 
            ],
            logfile='seg_water.log',
            raise_on_error=True
        )

        fat_glob = self.kwargs.get("fat_glob", "fat.nii.gz")
        if not fat_glob:
            LOG.info(" - No fat image specified, skipping SAT segmentation")
        else:
            fat = self.single_inimg(self._src_dir, fat_glob)
            if fat is None:
                self.no_data(f"No input image found in {self._src_dir}/{fat_glob}")
            if crop:
                fat = fat.reorient2std()
                fat_data_cropped = fat.data[:, :, zmin:]
                fat = fat.save_derived(fat_data_cropped, self.outfile("fat_cropped.nii.gz"))

            LOG.info(f" - Running TotalSeg for SAT using fat {fat.fname}")
            self.runcmd([
                    'TotalSegmentator',
                    '-i', fat.fpath,
                    '-o', self.outdir,
                    '--task', 'tissue_types_mr',
                ],
                logfile='seg_fat.log',
                raise_on_error=True
            )

        # Combined kidney mask
        kidney_left = self.single_inimg(self.name, "kidney_left.nii.gz", src=self.OUTPUT)
        kidney_right = self.single_inimg(self.name, "kidney_right.nii.gz", src=self.OUTPUT)
        kidney_combined = ((kidney_left.data > 0) | (kidney_right.data > 0)).astype(np.int8)
        kidney_left.save_derived(kidney_combined, self.outfile("kidneys.nii.gz"))

        # Generate overlays and calculate volumes
        segs_of_interest = self.kwargs.get("segs", None)
        nifti_files = glob.glob(self.outfile("*.nii.gz"))
        csv_path = self.outfile("volumes.csv")
        LOG.info(f" - Found {len(nifti_files)} segmentation files. Generating overlays and CSV at {csv_path}")

        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(geometryTolerance=1e-3)
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("shape")

        suffix = self.kwargs.get("csv_suffix", "")
        with open(csv_path, "w") as csv_file:
            for nifti_file in nifti_files:
                seg_img = self.single_inimg(self.name, nifti_file, src=self.OUTPUT)
                if segs_of_interest and seg_img.fname_noext not in segs_of_interest:
                    LOG.info(f" - Skipping {seg_img.fname_noext} - not in requested list")
                    continue

                volume = np.count_nonzero(seg_img.data) * seg_img.voxel_volume
                csv_file.write(f"{seg_img.fname_noext}{suffix}_vol,{volume}\n")
                LOG.info(f" - {seg_img.fname_noext}: volume = {volume} mL")

                if np.count_nonzero(seg_img.data) > 0:
                    try:
                        radiomics_results = extractor.execute(water.fpath, seg_img.fpath)
                        for k, v in radiomics_results.items():
                            if k.startswith("diagnostics"):
                                continue
                            elif "SurfaceArea" in k:
                                csv_file.write(f"{seg_img.fname_noext}{suffix}_sa,{v/100}\n")
                            elif "SurfaceVolumeRatio" in k:
                                csv_file.write(f"{seg_img.fname_noext}{suffix}_svr,{v*10}\n")
                    except Exception:
                        LOG.warn(f" - Radiomics extraction failed for {seg_img.fname_noext}, setting surface area and surface volume ratio to 0")
                        csv_file.write(f"{seg_img.fname_noext}{suffix}_sa,0\n")
                        csv_file.write(f"{seg_img.fname_noext}{suffix}_svr,0\n")
                else:
                    csv_file.write(f"{seg_img.fname_noext}{suffix}_sa,0\n")
                    csv_file.write(f"{seg_img.fname_noext}{suffix}_svr,0\n")

                # Generate overlay PNG
                overlay_name = f"{seg_img.fname_noext}_overlay"
                self.lightbox(water, seg_img, name=overlay_name, tight=True)

        dilate = self.kwargs.get("dilate", 0)
        if dilate:
            csv_dilated = csv_path.replace(".csv", "_dilated.csv")
            LOG.info(f" - Dilating segmentations by {dilate} voxels and saving to {csv_dilated}")
            with open(csv_dilated, "w") as csv_file:
                for nifti_file in nifti_files:
                    seg_img = self.single_inimg(self.name, nifti_file, src=self.OUTPUT)
                    if segs_of_interest and seg_img.fname_noext not in segs_of_interest:
                        LOG.info(f" - Skipping {seg_img.fname_noext} - not in requested list")
                        continue

                    dilated_data = binary_dilation(seg_img.data, iterations=dilate).astype(np.int8)
                    dilated_img = seg_img.save_derived(dilated_data, self.outfile(f"{seg_img.fname_noext}_dilated.nii.gz"))
                    volume = np.count_nonzero(dilated_data) * dilated_img.voxel_volume
                    csv_file.write(f"{dilated_img.fname_noext}{suffix},{volume}\n")
                    LOG.info(f" - {dilated_img.fname_noext}: dilated volume = {volume} mL")

                    if np.count_nonzero(dilated_img.data) > 0:
                        try:
                            radiomics_results = extractor.execute(water.fpath, dilated_img.fpath)
                            for k, v in radiomics_results.items():
                                if k.startswith("diagnostics"):
                                    continue
                                elif "SurfaceArea" in k:
                                    csv_file.write(f"{dilated_img.fname_noext}{suffix}_sa,{v/100}\n")
                                elif "SurfaceVolumeRatio" in k:
                                    csv_file.write(f"{dilated_img.fname_noext}{suffix}_svr,{v*10}\n")
                        except Exception:
                            LOG.warn(f" - Radiomics extraction failed for {dilated_img.fname_noext}, setting surface area and surface volume ratio to 0")
                            csv_file.write(f"{dilated_img.fname_noext}{suffix}_sa,0\n")
                            csv_file.write(f"{dilated_img.fname_noext}{suffix}_svr,0\n")
                    else:
                        csv_file.write(f"{dilated_img.fname_noext}{suffix}_sa,0\n")
                        csv_file.write(f"{dilated_img.fname_noext}{suffix}_svr,0\n")

                    # Generate overlay PNG for dilated segmentation
                    overlay_name = f"{dilated_img.fname_noext}_overlay"
                    self.lightbox(water, dilated_img, name=overlay_name, tight=True)

class TraceSeg(Module):
    def __init__(self, name="traceseg", **kwargs):
        self._src_dir = kwargs.get("src_dir", "t1")
        deps = [self._src_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        img_glob = self.kwargs.get("img_glob", "t1.nii.gz")
        img = self.single_inimg(self._src_dir, img_glob)
        if img is None:
            self.no_data(f"No input image found in {self._src_dir}/{img_glob}")
        LOG.info(f" - Running Trace Seg using {img.fname}")

        self.runcmd([
            'trace_segment',
            img.fpath,
            self.outfile(img.fname.replace('.nii.gz', '_seg.nii.gz')),
            ],
            logfile=f'trace_seg.log'
        )
        seg_img = ImageFile(self.outfile(img.fname.replace('.nii.gz', '_seg.nii.gz')), warn_json=False)
        with open(self.outfile("volumes.csv"), "w") as f:
            for idx, name in {
                1 : "trace_kidney_r",
                2 : "trace_kidney_l",
                3 : "trace_spleen",
                4 : "trace_liver", 
            }.items():
                roi = (seg_img.data == idx).astype(np.int8)
                organ_img = seg_img.save_derived(roi, self.outfile(f"{name}.nii.gz"))
                self.lightbox(img, organ_img, name=f"{name}_overlay", tight=True)
                volume = np.count_nonzero(roi) * seg_img.voxel_volume
                LOG.info(f" - {name}: volume = {volume} mL")
                f.write(f"{name},{volume}\n")
            tkv = np.logical_or(seg_img.data == 1, seg_img.data == 2)
            tvk_img = seg_img.save_derived(tkv.astype(np.int8), self.outfile("trace_kidney_all.nii.gz"))
            self.lightbox(img, tvk_img, name="trace_kidney_all_overlay", tight=True)
            tkv_volume = np.count_nonzero(tkv) * seg_img.voxel_volume
            LOG.info(f" - Total kidney volume: {tkv_volume} mL")
            f.write(f"trace_kidney_all,{tkv_volume}\n")


class OrganFat(Module):
    """
    Given organ segmentations split into fat/fat free portions
    """
    def __init__(self, name="seg_organ_fat", **kwargs):
        self._ff_dir = kwargs.get("ff_dir", "fat_fraction")
        self._seg_dir = kwargs.get("seg_dir", "seg_kidney_dixon")
        deps = [self._ff_dir, self._seg_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        ff_glob = self.kwargs.get("ff_glob", "fat_fraction.nii.gz")
        ff = self.single_inimg(self._ff_dir, ff_glob, src=self.OUTPUT)
        if ff is None:
            self.no_data(f"No fat fraction data found in {self._ff_dir}/{ff_glob}")

        seg_glob = self.kwargs.get("seg_glob", "kidney.nii.gz")
        segs = self.inimgs(self._seg_dir, seg_glob, src=self.OUTPUT)
        if segs is None:
            self.no_data(f"No segmentations found in {self._seg_dir}/{seg_glob}")

        ff_thresh = self.kwargs.get("ff_thresh", 15)
        for seg in segs:
            seg_filled = binary_fill_holes(seg.data)

            ff_data = self.resample(ff, seg, is_roi=False, allow_rotated=True).get_fdata()
            fat_mask = ff_data > ff_thresh

            LOG.info(f" - Segmenting {seg.fname} between fat / nofat using {ff.fname} with threshold {ff_thresh}")
            seg_nofat = np.copy(seg_filled)
            seg_nofat[fat_mask > 0] = 0

            seg_fat = np.copy(seg_filled)
            seg_fat[fat_mask == 0] = 0

            seg.save_derived(ff_data, self.outfile(f"{seg.fname_noext}_ff_res.nii.gz"))
            seg.save_derived(fat_mask.astype(np.int32), self.outfile(f"{seg.fname_noext}_fat_mask.nii.gz"))
            seg.save_derived(seg_nofat.astype(np.int32), self.outfile(f"{seg.fname_noext}_nofat.nii.gz"))
            seg.save_derived(seg_fat.astype(np.int32), self.outfile(f"{seg.fname_noext}_fat.nii.gz"))
