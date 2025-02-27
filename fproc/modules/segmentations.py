"""
FPROC: Segmentations of various body parts
"""
import logging
import validators
import wget

import numpy as np
import scipy
import skimage

from fsort import ImageFile
from fproc.module import Module

LOG = logging.getLogger(__name__)

class KneeToNeckDixon(Module):
    def __init__(self, name="seg_knee_to_neck_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        LOG.info(f" - Doing DIXON knee-to-nexck segmentation for subject")
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        src = self.kwargs.get("dixon_src", self.INPUT)
        knee_to_neck_model = self.pipeline.options.knee_to_neck_dixon_model
        fat = self.inimg(dixon_dir, "fat.nii.gz", src=src)
        t2star = self.inimg(dixon_dir, "t2star.nii.gz", src=src)
        water = self.inimg(dixon_dir, "water.nii.gz", src=src)

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
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        src = self.kwargs.get("dixon_src", self.INPUT)
        fat = self.inimg(dixon_dir, "fat.nii.gz", src=src)
        t2star = self.inimg(dixon_dir, "t2star.nii.gz", src=src)
        water = self.inimg(dixon_dir, "water.nii.gz", src=src)
        self.run_nnunetv2("14", [fat, t2star, water], "liver", "DIXON", water, "water")

class SpleenDixon(Module):
    def __init__(self, name="seg_spleen_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        src = self.kwargs.get("dixon_src", self.INPUT)
        fat = self.inimg(dixon_dir, "fat.nii.gz", src=src)
        t2star = self.inimg(dixon_dir, "t2star.nii.gz", src=src)
        water = self.inimg(dixon_dir, "water.nii.gz", src=src)
        self.run_nnunetv2("102", [fat, t2star, water], "spleen", "DIXON", water, "water")

class PancreasEthrive(Module):
    def __init__(self):
        Module.__init__(self, "seg_pancreas_ethrive")

    def process(self):
        ethrive = self.inimg("ethrive", "ethrive.nii.gz")
        self.run_nnunetv2("234", [ethrive], "pancreas", "eTHRIVE", ethrive, "ethrive")

class KidneyCystT2w(Module):
    def __init__(self, name="seg_kidney_cyst_t2w", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t2w_dir = self.kwargs.get("t2w_dir", "t2w")
        t2w_glob = self.kwargs.get("t2w_glob", "t2w.nii.gz")
        t2w_src = self.kwargs.get("t2w_src", self.INPUT)
        t2w_map = self.single_inimg(t2w_dir, t2w_glob, src=t2w_src)
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
    def __init__(self, name="seg_kidney_t1", map_dir="t1_kidney", map_glob="t1_map*.nii.gz"):
        Module.__init__(self, name)
        self._dir = map_dir
        self._glob = map_glob

    def process(self):
        t1_maps = self.inimgs(self._dir, self._glob, is_depfile=True)
        if not t1_maps:
            self.no_data(f"No T1 maps found to segment in {self._dir}/{self._glob}")

        single_map = len(t1_maps) == 1
        for t1_map in t1_maps:
            LOG.info(f" - Segmenting KIDNEY using T1 data: {t1_map.fname}")
            if single_map:
                out_prefix = "kidney"
                t1_map.save(self.outfile("t1_map.nii.gz"))
            else:
                out_prefix = f'kidney_{t1_map.fname_noext}'
                t1_map.save(self.outfile(t1_map.fname))
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
    def __init__(self, t2w_srcdir="t2w", **kwargs):
        Module.__init__(self, "seg_kidney_t2w", **kwargs)
        self.t2w_srcdir = t2w_srcdir

    def process(self):
        t2w_map = self.inimg(self.t2w_srcdir, "t2w.nii.gz")
        LOG.info(f" - Segmenting KIDNEY using T2w data: {t2w_map.fname}")

        model_weights = self.pipeline.options.kidney_t2w_model
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

class SatDixon(Module):
    def __init__(self, name="seg_sat_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        src = self.kwargs.get("dixon_src", self.INPUT)
        fat = self.inimg(dixon_dir, "fat.nii.gz", src=src)
        water = self.inimg(dixon_dir, "water.nii.gz", src=src)
        self.run_nnunetv2("141", [fat], "sat", "DIXON", water, "water")

class BodyDixon(Module):
    def __init__(self, name="seg_body_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        """
        Estimate body mask from dixon water map
        """
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        src = self.kwargs.get("dixon_src", self.INPUT)
        water_thresh = self.kwargs.get("water_thresh", 20)
        water = self.inimg(dixon_dir, "water.nii.gz", src=src)
        LOG.info(f" - Segmenting body by thresholding water map {water.fname} at level {water_thresh}")

        # Work slicewise and try to segment the body by thresholding, filling holes and
        # selecting the largest contiguous region (blob)
        water_data = water.data.squeeze()
        body_mask = []
        for sl in range(water.shape[2]):
            water_data_slice = water_data[..., sl]
            water_data_nonzero = water_data_slice[water_data_slice > 0]
            thresh = np.percentile(water_data_nonzero, water_thresh)
            mask_slice = (water_data_slice > thresh).astype(np.int8).squeeze()
            mask_slice_filled = scipy.ndimage.morphology.binary_fill_holes(mask_slice)
            largest_blob = self.blobs_by_size(mask_slice_filled, min_size=10)[0]
            body_mask.append(largest_blob)
        body_mask = np.stack(body_mask, axis=-1)

        water.save_derived(body_mask, self.outfile("body.nii.gz"))
        seg = self.inimg(self.name, "body.nii.gz", src=self.OUTPUT)
        self.lightbox(water, seg, name="body_water_lightbox", tight=True)

class VatDixon(Module):
    def __init__(self, name="seg_vat_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        """
        Get VAT from all fat - (SAT + organ masks)
        """
        ff_dir = self.kwargs.get("ff_dir", "fat_fraction")
        ff_src = self.kwargs.get("ff_src", self.OUTPUT)
        ff_thresh = self.kwargs.get("ff_thresh", 85)
        ff_glob = self.kwargs.get("ff_glob", "fat_fraction.nii.gz")
        ff = self.inimg(ff_dir, ff_glob, src=ff_src)
        organs = self.kwargs.get("organs", {})
        vat_data = (ff.data > ff_thresh).astype(np.int8)
        while vat_data.ndim > 3:
            vat_data = vat_data.squeeze(-1)

        body = self.inimg("seg_body_dixon", "body.nii.gz", src=self.OUTPUT)
        vat_data[body.data == 0] = 0
        ff.save_derived(vat_data, self.outfile("fat.nii.gz"))

        sat = self.inimg("seg_sat_dixon", "sat.nii.gz", src=self.OUTPUT)
        vat_data[sat.data > 0] = 0
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

        # FIXME temporary remove top/bottom slices to avoid problem with SAT segmentor
        vat_data[..., 0] = 0
        vat_data[..., -1] = 0

        ff.save_derived(vat_data, self.outfile("vat.nii.gz"))
        seg = self.inimg(self.name, "vat.nii.gz", src=self.OUTPUT)
        
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        dixon_src = self.kwargs.get("dixon_src", self.INPUT)
        water = self.inimg(dixon_dir, "water.nii.gz", src=dixon_src)
        if water is not None:
            self.lightbox(water, seg, name="vat_water_lightbox", tight=True)
        else:
            LOG.warn(" - No water map found to overlay with VAT")

class LegDixon(Module):
    def __init__(self, name="seg_leg_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        src = self.kwargs.get("dixon_src", self.INPUT)
        water = self.inimg(dixon_dir, "water.nii.gz", src=src)
        fat = self.inimg(dixon_dir, "fat.nii.gz", src=src)

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

class KidneyDixon(Module):
    def __init__(self, name="seg_kidney_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        src = self.kwargs.get("dixon_src", self.INPUT)
        fat = self.inimg(dixon_dir, "fat.nii.gz", src=src)
        ff =self.inimg(dixon_dir, "fat_fraction.nii.gz", src=src)
        t2star = self.inimg(dixon_dir, "t2star.nii.gz", src=src)
        water = self.inimg(dixon_dir, "water.nii.gz", src=src)
        self.run_nnunetv2("326", [fat, ff, t2star, water], "kidney", "DIXON", water, "water")

