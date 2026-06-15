# Full wbdixon pipeline from XNAT dicom data
#
# includes stitching and classification of dixon data
#
# We have two parallel branches for the dixon data, one
# with bias correction and normalization during stitching
# and one with no preprocessing.
#
# The bias-corrected and normalized branch is used for the
# segmentation, while the unprocessed branch is used for
# the quantitative maps.

from glob import glob
import logging
import os

from importlib.resources import path
import numpy as np

from fproc.module import Module
from fproc.modules import maps, statistics, regrid, segmentations, seg_postprocess

LOG = logging.getLogger(__name__)

__version__ = "0.0.1"

NAME = "wbdixon_full"

MUSCLE_MAP_ROIS = {
    1101: "left_levator_scapulae",
    1102: "right_levator_scapulae",
    1111: "left_semispinalis_cervicis_and_multifidus",
    1112: "right_semispinalis_cervicis_and_multifidus",
    1121: "left_semispinalis_capitis",
    1122: "right_semispinalis_capitis",
    1131: "left_splenius_capitis",
    1132: "right_splenius_capitis",
    1141: "left_sternocleidomastoid",
    1142: "right_sternocleidomastoid",
    1151: "left_longus_colli",
    1152: "right_longus_colli",
    1161: "left_trapezius",
    1162: "right_trapezius",
    2101: "left_supraspinatus",
    2102: "right_supraspinatus",
    2111: "left_subscapularis",
    2112: "right_subscapularis",
    2121: "left_infraspinatus",
    2122: "right_infraspinatus",
    2141: "left_deltoid",
    2142: "right_deltoid",
    4101: "left_rhomboid",
    4102: "right_rhomboid",
    5101: "left_thoracolumbar_multifidus",
    5102: "right_thoracolumbar_multifidus",
    5111: "left_erector_spinae",
    5112: "right_erector_spinae",
    5121: "left_psoas_major",
    5122: "right_psoas_major",
    5131: "left_quadratus_lumborum",
    5132: "right_quadratus_lumborum",
    5141: "left_lattisimus_dorsi",
    5142: "right_lattisimus_dorsi",
    6101: "left_gluteus_minimus",
    6102: "right_gluteus_minimus",
    6111: "left_gluteus_medius",
    6112: "right_gluteus_medius",
    6121: "left_gluteus_maximus",
    6122: "right_gluteus_maximus",
    6131: "left_tensor_fascia_latae",
    6132: "right_tensor_fascia_latae",
    6141: "left_iliacus",
    6142: "right_iliacus",
    6151: "left_ilium",
    6152: "right_ilium",
    6160: "sacrum",
    6171: "left_femur",
    6172: "right_femur",
    6181: "left_piriformis",
    6182: "right_piriformis",
    6191: "left_pectineus",
    6192: "right_pectineus",
    6201: "left_obturator_internus",
    6202: "right_obturator_internus",
    6211: "left_obturator_externus",
    6212: "right_obturator_externus",
    6221: "left_gemelli_and_quadratus_femoris",
    6222: "right_gemelli_and_quadratus_femoris",
    7101: "left_vastus_lateralis",
    7102: "right_vastus_lateralis",
    7111: "left_vastus_intermedius",
    7112: "right_vastus_intermedius",
    7121: "left_vastus_medialis",
    7122: "right_vastus_medialis",
    7131: "left_rectus_femoris",
    7132: "right_rectus_femoris",
    7141: "left_sartorius",
    7142: "right_sartorius",
    7151: "left_gracilis",
    7152: "right_gracilis",
    7161: "left_semimembranosus",
    7162: "right_semimembranosus",
    7171: "left_semitendinosus",
    7172: "right_semitendinosus",
    7181: "left_biceps_femoris_long_head",
    7182: "right_biceps_femoris_long_head",
    7191: "left_biceps_femoris_short_head",
    7192: "right_biceps_femoris_short_head",
    7201: "left_adductor_magnus",
    7202: "right_adductor_magnus",
    7211: "left_adductor_longus",
    7212: "right_adductor_longus",
    7221: "left_adductor_brevis",
    7222: "right_adductor_brevis",
    8101: "left_anterior_compartment",
    8102: "right_anterior_compartment",
    8111: "left_deep_posterior_compartment",
    8112: "right_deep_posterior_compartment",
    8121: "left_lateral_compartment",
    8122: "right_lateral_compartment",
    8131: "left_soleus",
    8132: "right_soleus",
    8141: "left_gastrocnemius",
    8142: "right_gastrocnemius",
    8151: "left_tibia",
    8152: "right_tibia",
    8161: "left_fibula",
    8162: "right_fibula",
}


class BiasCorr(Module):
    def __init__(self, name="biascorr", **kwargs):
        self._src_dir = kwargs.get("src_dir", "raw_dixon")
        Module.__init__(self, name, deps=[self._src_dir], **kwargs)

    def process(self):
        src_glob = self.kwargs.get("src_glob", "raw_dixon*.nii.gz")
        imgs = self.inimgs(self._src_dir, src_glob)

        for img in imgs:
            LOG.info(f"Processing {img.fname}")
            img_data = img.data
            original_ndim = img.ndim

            while len(img_data.shape) < 4:
                img_data = np.expand_dims(img_data, axis=-1)

            LOG.info(
                f" - Image with {img_data.shape[3]} volumes, processing each volume separately"
            )
            corrected_vols = []

            for vol_idx in range(img_data.shape[3]):
                LOG.info(f" - Processing volume {vol_idx + 1}/{img_data.shape[3]}")
                vol_path = self.outfile(f"{img.fname_noext}_vol{vol_idx}.nii.gz")
                img.save_derived(img_data[..., vol_idx], vol_path)

                # Run bias correction on single volume
                retval = self.runcmd(
                    [
                        "fast",
                        "-n",
                        "4",
                        "-H",
                        "0.1",
                        "-I",
                        "4",
                        "-l",
                        "20.0",
                        "-B",
                        "-o",
                        self.outfile(f"{img.fname_noext}_vol{vol_idx}_biascorr"),
                        vol_path,
                    ],
                    logfile=f"{img.fname_noext}_vol{vol_idx}_biascorr.log",
                    raise_on_error=False,
                )

                if retval != 0:
                    LOG.warning(
                        f" - Bias correction failed for volume {vol_idx + 1} of {img.fname} - using original volume for output"
                    )
                    corrected_vols.append(img_data[..., vol_idx])
                else:
                    # Load corrected volume
                    corrected_vol = self.inimg(
                        self.name,
                        f"{img.fname_noext}_vol{vol_idx}_biascorr_restore.nii.gz",
                        src=self.OUTPUT,
                    )
                    data = np.copy(corrected_vol.data)
                    data[~np.isfinite(data)] = 0  # Replace NaNs and infs with 0
                    if np.all(data == 0):
                        LOG.warning(
                            f" - Corrected volume {vol_idx + 1} of {img.fname} is all zeros after processing - using original volume for output"
                        )
                        corrected_vols.append(img_data[..., vol_idx])
                    else:
                        corrected_vols.append(data)

                # Remove temporary files
                os.remove(vol_path)
                for path in glob(
                    self.outfile(f"{img.fname_noext}_vol{vol_idx}_biascorr*.nii.gz")
                ):
                    os.remove(path)

            # Stack volumes back into 4D
            LOG.info(f" - Stacking {len(corrected_vols)} corrected volumes")
            stacked_data = np.stack(corrected_vols, axis=-1)

            # Squeeze back to original dimensionality
            if original_ndim == 3:
                stacked_data = np.squeeze(stacked_data, axis=-1)

            output_path = self.outfile(f"{img.fname_noext}_biascorr_restore.nii.gz")
            img.save_derived(stacked_data, output_path)
            LOG.info(f" - Saved stacked output to {output_path}")


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


class FinalDixon(Module):
    def __init__(self, name="final_dixon", **kwargs):
        self._water_src = kwargs.get("water", "dixon_stitched")
        self._fat_src = kwargs.get("fat", "dixon_stitched")
        self._ff_src = kwargs.get("fat_fraction", "dixon_stitched")
        self._t2star_src = kwargs.get("t2star", "dixon_stitched")
        Module.__init__(
            self,
            name,
            deps=[self._water_src, self._fat_src, self._ff_src, self._t2star_src],
            **kwargs,
        )

    def process(self):
        water_img = self.single_inimg(self._water_src, "water.nii.gz")
        fat_img = self.single_inimg(self._fat_src, "fat.nii.gz")
        ff_img = self.single_inimg(self._ff_src, "fat_fraction.nii.gz")
        t2star_img = self.single_inimg(self._t2star_src, "t2star.nii.gz")
        for img in [water_img, fat_img, ff_img, t2star_img]:
            if img is None:
                LOG.error(f" - Missing input for FinalDixon")
                continue
            img.save(self.outfile(img.fname))


MODULES = [
    regrid.Stitch(
        name="dixon_stitched",
        img_dir="raw_dixon",
        imgs={
            "raw_dixon_series_?_1.nii.gz": "raw_dixon_1.nii.gz",
            "raw_dixon_series_?_2.nii.gz": "raw_dixon_2.nii.gz",
            "raw_dixon_series_?_3.nii.gz": "raw_dixon_3.nii.gz",
            "raw_dixon_series_?.nii.gz": "raw_dixon.nii.gz",
        },
        normalise=False,
    ),
    BiasCorr(
        name="dixon_stitched_biascorr",
        src_dir="dixon_stitched",
        src_glob="raw_dixon*.nii.gz",
    ),
    maps.DixonClassify(
        name="dixon_classify_biascorr",
        model="/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier_20250626.h5",
        fixes="/spmstore/project/RenalMRI/wbdixon_full/dixon_classify_fix.csv",
        dixon_src="dixon_stitched_biascorr",
        dixon_glob="raw_dixon*_biascorr_restore.nii.gz",
    ),
    maps.DixonClassify(
        name="dixon_classify",
        model="/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier_20250626.h5",
        fixes="/spmstore/project/RenalMRI/wbdixon_full/dixon_classify_fix.csv",
        dixon_src="dixon_stitched",
        dixon_glob="raw_dixon*.nii.gz",
    ),
    FinalDixon(
        name="dixon_final",
        water="dixon_classify_biascorr",
        fat="dixon_classify_biascorr",
        fat_fraction="dixon_classify",
        t2star="dixon_classify",
    ),
    maps.FatFractionDixon(dixon_dir="dixon_final", ff_name="fat_fraction"),
    maps.T2starDixon(dixon_dir="dixon_final", t2star_name="t2star"),
    RoiZ(name="dixon_final_roiz", src_dir="dixon_final", proportion=66),
    # Segmentations
    segmentations.LegDixon(dixon_dir="dixon_final"),
    segmentations.TotalSeg(src_dir="dixon_final", img_glob="water.nii.gz"),
    segmentations.LegDixonUsingTotalsegFemur(
        name="seg_leg_dixon_femur",
        dixon_dir="dixon_final",
        totalseg_dir="totalseg",
        largest_blob_only=True,
        dilate_muscle_masks=True,
    ),
    segmentations.MuscleMap(
        src_dir="dixon_final",
        src_glob="water.nii.gz",
    ),
    segmentations.MuscleMap(
        name="muscle_map_nobiascorr",
        src_dir="dixon_classify",
        src_glob="water.nii.gz",
    ),
    # Seg fixes
    seg_postprocess.SegFix(
        seg_dir="seg_leg_dixon",
        fix_dir_option="seg_leg_dixon_fix",
        segs={
            "calf_muscle_l.nii.gz": "%s/calf_muscle_l_fixed.nii.gz",
            "calf_muscle_r.nii.gz": "%s/calf_muscle_r_fixed.nii.gz",
            "thigh_muscle_l.nii.gz": "%s/thigh_muscle_l_fixed.nii.gz",
            "thigh_muscle_r.nii.gz": "%s/thigh_muscle_r_fixed.nii.gz",
            "calf_sat_l.nii.gz": "%s/calf_sat_l_fixed.nii.gz",
            "calf_sat_r.nii.gz": "%s/calf_sat_r_fixed.nii.gz",
            "thigh_sat_l.nii.gz": "%s/thigh_sat_l_fixed.nii.gz",
            "thigh_sat_r.nii.gz": "%s/thigh_sat_r_fixed.nii.gz",
        },
        map_dir="seg_leg_dixon",
        map_fname="water.nii.gz",
    ),
    # Statistics
    statistics.AllRoiStats(
        name="muscle_map_stats_nobiascorr",
        roi_dir="muscle_map_nobiascorr",
        roi_glob="*_dseg.nii.gz",
        roi_names=MUSCLE_MAP_ROIS,
        out_name="muscle_map_stats.csv",
    ),
    statistics.AllRoiStats(
        name="muscle_map_stats",
        roi_dir="muscle_map",
        roi_glob="*_dseg.nii.gz",
        roi_names=MUSCLE_MAP_ROIS,
        out_name="muscle_map_stats.csv",
    ),
    statistics.SegStats(
        name="stats",
        segs={
            "calf_muscle_r": {
                "dir": "seg_leg_dixon",
                "glob": "calf_muscle_r_nodil.nii.gz",
            },
            "calf_muscle_l": {
                "dir": "seg_leg_dixon",
                "glob": "calf_muscle_l_nodil.nii.gz",
            },
            "thigh_muscle_r": {
                "dir": "seg_leg_dixon",
                "glob": "thigh_muscle_r_nodil.nii.gz",
            },
            "thigh_muscle_l": {
                "dir": "seg_leg_dixon",
                "glob": "thigh_muscle_l_nodil.nii.gz",
            },
            "calf_sat_r": {"dir": "seg_leg_dixon", "glob": "calf_sat_r.nii.gz"},
            "calf_sat_l": {"dir": "seg_leg_dixon", "glob": "calf_sat_l.nii.gz"},
            "thigh_sat_r": {"dir": "seg_leg_dixon", "glob": "thigh_sat_r.nii.gz"},
            "thigh_sat_l": {"dir": "seg_leg_dixon", "glob": "thigh_sat_l.nii.gz"},
            "calf_muscle": {"dir": "seg_leg_dixon", "glob": "calf_muscle_nodil.nii.gz"},
            "calf_sat": {"dir": "seg_leg_dixon", "glob": "calf_sat.nii.gz"},
            "thigh_muscle": {
                "dir": "seg_leg_dixon",
                "glob": "thigh_muscle_nodil.nii.gz",
            },
            "thigh_sat": {"dir": "seg_leg_dixon", "glob": "thigh_sat.nii.gz"},
            "muscle_r": {"dir": "seg_leg_dixon", "glob": "muscle_r_nodil.nii.gz"},
            "sat_r": {"dir": "seg_leg_dixon", "glob": "sat_r.nii.gz"},
            "muscle_l": {"dir": "seg_leg_dixon", "glob": "muscle_l_nodil.nii.gz"},
            "sat_l": {"dir": "seg_leg_dixon", "glob": "sat_l.nii.gz"},
            "total": {"dir": "seg_leg_dixon", "glob": "total.nii.gz"},
            "pancreas": {
                "dir": "totalseg",
                "glob": "pancreas.nii.gz",
                "params": ["ff_calc"],
            },
            "liver": {"dir": "totalseg", "glob": "liver.nii.gz", "params": ["ff_calc"]},
        },
        params={
            "ff_scanner": {
                "dir": "fat_fraction",
                "glob": "fat_fraction_scanner.nii.gz",
                "limits": (0, 100),
            },
            "ff_calc": {
                "dir": "fat_fraction",
                "glob": "fat_fraction_calc.nii.gz",
                "limits": (0, 100),
            },
            "t2star": {
                "dir": "t2star_dixon",
                "glob": "t2star_exclude_fill.nii.gz",
                "limits": (2, 100),
            },
        },
        stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
        seg_volumes=True,
    ),
    # Statistics
    statistics.SegStats(
        name="stats_newleg",
        segs={
            "calf_muscle_r": {
                "dir": "seg_leg_dixon_femur",
                "glob": "calf_muscle_r_nodil.nii.gz",
            },
            "calf_muscle_l": {
                "dir": "seg_leg_dixon_femur",
                "glob": "calf_muscle_l_nodil.nii.gz",
            },
            "thigh_muscle_r": {
                "dir": "seg_leg_dixon_femur",
                "glob": "thigh_muscle_r_nodil.nii.gz",
            },
            "thigh_muscle_l": {
                "dir": "seg_leg_dixon_femur",
                "glob": "thigh_muscle_l_nodil.nii.gz",
            },
            "calf_sat_r": {"dir": "seg_leg_dixon_femur", "glob": "calf_sat_r.nii.gz"},
            "calf_sat_l": {"dir": "seg_leg_dixon_femur", "glob": "calf_sat_l.nii.gz"},
            "thigh_sat_r": {"dir": "seg_leg_dixon_femur", "glob": "thigh_sat_r.nii.gz"},
            "thigh_sat_l": {"dir": "seg_leg_dixon_femur", "glob": "thigh_sat_l.nii.gz"},
            "calf_muscle": {
                "dir": "seg_leg_dixon_femur",
                "glob": "calf_muscle_nodil.nii.gz",
            },
            "calf_sat": {"dir": "seg_leg_dixon_femur", "glob": "calf_sat.nii.gz"},
            "thigh_muscle": {
                "dir": "seg_leg_dixon_femur",
                "glob": "thigh_muscle_nodil.nii.gz",
            },
            "thigh_sat": {"dir": "seg_leg_dixon_femur", "glob": "thigh_sat.nii.gz"},
            "muscle_r": {"dir": "seg_leg_dixon_femur", "glob": "muscle_r_nodil.nii.gz"},
            "sat_r": {"dir": "seg_leg_dixon_femur", "glob": "sat_r.nii.gz"},
            "muscle_l": {"dir": "seg_leg_dixon_femur", "glob": "muscle_l_nodil.nii.gz"},
            "sat_l": {"dir": "seg_leg_dixon_femur", "glob": "sat_l.nii.gz"},
            "total": {"dir": "seg_leg_dixon_femur", "glob": "total_nodil.nii.gz"},
            "pancreas": {
                "dir": "totalseg",
                "glob": "pancreas.nii.gz",
                "params": ["ff_calc"],
            },
            "liver": {"dir": "totalseg", "glob": "liver.nii.gz", "params": ["ff_calc"]},
        },
        params={
            "ff_scanner": {
                "dir": "fat_fraction",
                "glob": "fat_fraction_scanner.nii.gz",
                "limits": (0, 100),
            },
            "ff_calc": {
                "dir": "fat_fraction",
                "glob": "fat_fraction_calc.nii.gz",
                "limits": (0, 100),
            },
            "t2star": {
                "dir": "t2star_dixon",
                "glob": "t2star_exclude_fill.nii.gz",
                "limits": (2, 100),
            },
        },
        stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
        seg_volumes=True,
    ),
    statistics.SegStats(
        name="stats_newleg_dil",
        segs={
            "calf_muscle_r": {
                "dir": "seg_leg_dixon_femur",
                "glob": "calf_muscle_r.nii.gz",
            },
            "calf_muscle_l": {
                "dir": "seg_leg_dixon_femur",
                "glob": "calf_muscle_l.nii.gz",
            },
            "thigh_muscle_r": {
                "dir": "seg_leg_dixon_femur",
                "glob": "thigh_muscle_r.nii.gz",
            },
            "thigh_muscle_l": {
                "dir": "seg_leg_dixon_femur",
                "glob": "thigh_muscle_l.nii.gz",
            },
            "calf_sat_r": {"dir": "seg_leg_dixon_femur", "glob": "calf_sat_r.nii.gz"},
            "calf_sat_l": {"dir": "seg_leg_dixon_femur", "glob": "calf_sat_l.nii.gz"},
            "thigh_sat_r": {"dir": "seg_leg_dixon_femur", "glob": "thigh_sat_r.nii.gz"},
            "thigh_sat_l": {"dir": "seg_leg_dixon_femur", "glob": "thigh_sat_l.nii.gz"},
            "calf_muscle": {"dir": "seg_leg_dixon_femur", "glob": "calf_muscle.nii.gz"},
            "calf_sat": {"dir": "seg_leg_dixon_femur", "glob": "calf_sat.nii.gz"},
            "thigh_muscle": {
                "dir": "seg_leg_dixon_femur",
                "glob": "thigh_muscle.nii.gz",
            },
            "thigh_sat": {"dir": "seg_leg_dixon_femur", "glob": "thigh_sat.nii.gz"},
            "muscle_r": {"dir": "seg_leg_dixon_femur", "glob": "muscle_r.nii.gz"},
            "sat_r": {"dir": "seg_leg_dixon_femur", "glob": "sat_r.nii.gz"},
            "muscle_l": {"dir": "seg_leg_dixon_femur", "glob": "muscle_l.nii.gz"},
            "sat_l": {"dir": "seg_leg_dixon_femur", "glob": "sat_l.nii.gz"},
            "total": {"dir": "seg_leg_dixon_femur", "glob": "total.nii.gz"},
            "pancreas": {
                "dir": "totalseg",
                "glob": "pancreas.nii.gz",
                "params": ["ff_calc"],
            },
            "liver": {"dir": "totalseg", "glob": "liver.nii.gz", "params": ["ff_calc"]},
        },
        params={
            "ff_scanner": {
                "dir": "fat_fraction",
                "glob": "fat_fraction_scanner.nii.gz",
                "limits": (0, 100),
            },
            "ff_calc": {
                "dir": "fat_fraction",
                "glob": "fat_fraction_calc.nii.gz",
                "limits": (0, 100),
            },
            "t2star": {
                "dir": "t2star_dixon",
                "glob": "t2star_exclude_fill.nii.gz",
                "limits": (2, 100),
            },
        },
        stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
        seg_volumes=True,
    ),
]


def add_options(parser):
    parser.add_argument(
        "--seg-leg-dixon-fix", help="Directory containing fixed leg masks"
    )
