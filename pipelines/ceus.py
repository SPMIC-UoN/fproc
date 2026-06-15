# CEUS pipeline
# cor and ax T1s, everything taking from kidney (cor) but should use ax for liver, pancreas

import logging
import os

import numpy as np

from fsort.image_file import ImageFile
from fproc.module import Module
from fproc.modules import (
    maps,
    segmentations,
    statistics,
    seg_postprocess,
    regrid,
    align,
    misc,
)

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME = "ceus"


class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats",
            default_limits="3t",
            segs={
                "kidney_cortex": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*cortex*.nii.gz",
                },
                "kidney_cortex_l": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*cortex_l*.nii.gz",
                },
                "kidney_cortex_r": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*cortex_r*.nii.gz",
                },
                "kidney_medulla": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*medulla*.nii.gz",
                },
                "kidney_medulla_l": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*medulla_l*.nii.gz",
                },
                "kidney_medulla_r": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*medulla_r*.nii.gz",
                },
                "kidney_l": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*_l.nii.gz",
                    "params": ["b1_stim"],
                },
                "kidney_r": {
                    "dir": "seg_kidney_t1_clean",
                    "glob": "*_r.nii.gz",
                    "params": ["b1_stim"],
                },
                "tkv_l": {
                    "dir": "seg_kidney_t2w_fix",
                    "glob": "*left*.nii.gz",
                },
                "tkv_r": {
                    "dir": "seg_kidney_t2w_fix",
                    "glob": "*right*.nii.gz",
                },
            },
            params={
                "t2star_exp": {
                    "dir": "t2star",
                    "glob": "t2star_2p_exp*.nii.gz",
                },
                "t2star_loglin": {
                    "dir": "t2star",
                    "glob": "t2star_loglin*.nii.gz",
                },
                "r2star_exp": {
                    "dir": "t2star",
                    "glob": "r2star_2p_exp*.nii.gz",
                },
                "r2star_loglin": {
                    "dir": "t2star",
                    "glob": "r2star_loglin*.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_cor",
                    "glob": "t1_conf.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_clean_native"},
                    },
                },
                "t1_molli_nomdr": {
                    "dir": "t1_molli_nomdr",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_clean_native"},
                    },
                },
                "t1_molli_mdr": {
                    "dir": "t1_molli_mdr",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_mdr_clean_native"},
                    },
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                    "segs": ["tkv_l", "tkv_r"],
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                    "segs": ["tkv_l", "tkv_r"],
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                    "segs": ["tkv_l", "tkv_r"],
                },
                "ff": {
                    "dir": "ff_dixon",
                    "glob": "fat_fraction.nii.gz",
                },
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd", "perc90", "te", "mode", "fwhm"],
        )


class StatsDixon(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats_dixon",
            default_limits="3t",
            segs={
                "liver": {
                    "dir": "seg_liver_dixon",
                    "glob": "liver.nii.gz",
                    "seg_volumes": False,
                },
                "spleen": {
                    "dir": "seg_spleen_dixon",
                    "glob": "spleen.nii.gz",
                    "seg_volumes": False,
                },
                "sat": {
                    "dir": "seg_sat_dixon",
                    "glob": "sat.nii.gz",
                    "seg_volumes": False,
                },
                "vat": {
                    "dir": "seg_vat_dixon_local",
                    "glob": "vat.nii.gz",
                    "params": [],  # Volumes only
                },
                "pancreas": {
                    "dir": "seg_pancreas_ethrive",
                    "glob": "pancreas.nii.gz",
                },
                "kidney_dixon_nofat": {
                    "dir": "seg_kidney_fat_dixon",
                    "glob": "kidney_parenchyma.nii.gz",
                    "params": [],
                },
                "kidney_dixon_left_nofat": {
                    "dir": "seg_kidney_fat_dixon",
                    "glob": "kidney_parenchyma_left.nii.gz",
                    "params": [],
                },
                "kidney_dixon_right_nofat": {
                    "dir": "seg_kidney_fat_dixon",
                    "glob": "kidney_parenchyma_right.nii.gz",
                    "params": [],
                },
                "fat_pelvis": {
                    "dir": "seg_kidney_fat_dixon",
                    "glob": "fat_pelvis.nii.gz",
                    "params": ["ff"],
                },
                "fat_pelvis_left": {
                    "dir": "seg_kidney_fat_dixon",
                    "glob": "fat_pelvis_left.nii.gz",
                    "params": ["ff"],
                },
                "fat_pelvis_right": {
                    "dir": "seg_kidney_fat_dixon",
                    "glob": "fat_pelvis_right.nii.gz",
                    "params": ["ff"],
                },
            },
            params={
                "t2star": {
                    "dir": "t2star_dixon",
                    "glob": "t2star.nii.gz",
                },
                "r2star": {
                    "dir": "t2star_dixon",
                    "glob": "r2star_t2star.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_cor",
                    "glob": "t1_conf.nii.gz",
                    "segs" : [
                        "spleen",
                        "sat",
                        "kidney_dixon_nofat",
                        "kidney_dixon_left_nofat",
                        "kidney_dixon_right_nofat",
                        "fat_pelvis",
                        "fat_pelvis_left",
                        "fat_pelvis_right"
                    ],
                },
                "t1_ax": {
                    "dir": "t1_molli_ax",
                    "glob": "t1_conf.nii.gz",
                    "segs" : [
                        "liver",
                        "pancreas",
                    ]
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                },
                "ff": {
                    "dir": "ff_dixon",
                    "glob": "fat_fraction.nii.gz",
                },
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd", "perc90", "te", "mode", "fwhm"],
            seg_volumes=True,
        )

class StatsDixonTotalseg(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats_dixon_totalseg",
            default_limits="3t",
            segs={
                "liver": {
                    "dir": "totalseg",
                    "glob": "liver.nii.gz",
                    "seg_volumes": False,
                },
                "spleen": {
                    "dir": "totalseg",
                    "glob": "spleen.nii.gz",
                    "seg_volumes": False,
                },
                "sat": {
                    "dir": "totalseg",
                    "glob": "subcutaneous_fat.nii.gz",
                    "seg_volumes": False,
                },
                "vat": {
                    "dir": "seg_vat_dixon_totalseg",
                    "glob": "vat.nii.gz",
                    "params": [],  # Volumes only
                },
                "pancreas": {
                    "dir": "totalseg",
                    "glob": "pancreas.nii.gz",
                },
                "kidney_dixon_nofat": {
                    "dir": "seg_kidney_fat_dixon_totalseg",
                    "glob": "kidney_parenchyma.nii.gz",
                    "params": [],
                },
                "kidney_dixon_left_nofat": {
                    "dir": "seg_kidney_fat_dixon_totalseg",
                    "glob": "kidney_parenchyma_left.nii.gz",
                    "params": [],
                },
                "kidney_dixon_right_nofat": {
                    "dir": "seg_kidney_fat_dixon_totalseg",
                    "glob": "kidney_parenchyma_right.nii.gz",
                    "params": [],
                },
                "fat_pelvis": {
                    "dir": "seg_kidney_fat_dixon_totalseg",
                    "glob": "fat_pelvis.nii.gz",
                    "params": ["ff"],
                },
                "fat_pelvis_left": {
                    "dir": "seg_kidney_fat_dixon_totalseg",
                    "glob": "fat_pelvis_left.nii.gz",
                    "params": ["ff"],
                },
                "fat_pelvis_right": {
                    "dir": "seg_kidney_fat_dixon_totalseg",
                    "glob": "fat_pelvis_right.nii.gz",
                    "params": ["ff"],
                },
            },
            params={
                "t2star": {
                    "dir": "t2star_dixon",
                    "glob": "t2star.nii.gz",
                },
                "r2star": {
                    "dir": "t2star_dixon",
                    "glob": "r2star_t2star.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_cor",
                    "glob": "t1_conf.nii.gz",
                    "segs" : [
                        "spleen",
                        "sat",
                        "kidney_dixon_nofat",
                        "kidney_dixon_left_nofat",
                        "kidney_dixon_right_nofat",
                        "fat_pelvis",
                        "fat_pelvis_left",
                        "fat_pelvis_right"
                    ]
                },
                "t1_ax": {
                    "dir": "t1_molli_ax",
                    "glob": "t1_conf.nii.gz",
                    "segs" : [
                        "liver",
                        "pancreas",
                    ]
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                },
                "ff": {
                    "dir": "ff_dixon",
                    "glob": "fat_fraction.nii.gz",
                },
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd", "perc90", "te", "mode", "fwhm"],
            seg_volumes=True,
        )



class T1MolliMetadata(Module):
    def __init__(self, name="t1_molli_md", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t1_dir = "t1_molli_cor"
        t1_glob = "t1_*.nii.gz"
        t1s = self.inimgs(t1_dir, t1_glob, src=self.INPUT)
        if not t1s:
            self.no_data(f"No T1 maps found in {t1_dir} matching {t1_glob}")

        tis = []
        hr = []
        for t1 in t1s:
            tis.extend(list(t1.inversiontimedelay))
            hr.extend(list(t1.heartrate))

        hr = np.unique(hr)
        if len(hr) > 1:
            LOG.warn(f"Multiple heart rates found: {hr} - using first")
            hr = hr[0]
        elif len(hr) == 0:
            LOG.warn("No heart rate found")
            hr = ""
        else:
            hr = hr[0]
            LOG.info(f" - Found heart rate: {hr}")

        tis = sorted([float(v) for v in np.unique(tis) if float(v) > 0])
        with open(self.outfile("tis.txt"), "w") as f:
            f.write("\n".join([str(v) for v in tis]))

        LOG.info(f" - Found TIs: {tis}")
        if len(tis) >= 3:
            ti1, ti2, spacing = tis[0], tis[1], tis[2] - tis[0]
        else:
            ti1, ti2, spacing = "", "", ""
            LOG.warn(f"Not enough TIs found: {tis}")

        with open(self.outfile("t1_molli_md.csv"), "w") as f:
            f.write(f"t1_molli_heart_rate,{hr}\n")
            f.write(f"t1_molli_ti1,{ti1}\n")
            f.write(f"t1_molli_ti2,{ti2}\n")
            f.write(f"t1_molli_ti_spacing,{spacing}\n")


class T1Scaled(Module):

    def __init__(self, name="t1_scaled", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        scale_factors_fname = self.pipeline.options.t1_scale_factors
        sf_left, sf_right = None, None
        if scale_factors_fname and os.path.isfile(scale_factors_fname):
            with open(scale_factors_fname) as f:
                for l in f.readlines()[1:]:
                    parts = l.split(",")
                    if len(parts) != 3:
                        LOG.warn(f"Invalid line in scale factors file: {l}")
                        continue
                    subjid = parts[0]
                    if (
                        subjid.strip().lower()
                        == self.pipeline.options.subjid.strip().lower()
                    ):
                        sf_left, sf_right = float(parts[1]), float(parts[2])
                        break

        if not sf_left:
            LOG.warn(
                f"Scale factor (left) not found for {self.pipeline.options.subjid} - will output unscaled data"
            )
            sf_left = 1
        if not sf_right:
            LOG.warn(
                f"Scale factor (right) not found for {self.pipeline.options.subjid} - will output unscaled data"
            )
            sf_right = 1

        t1_dir = "t1_molli_cor"
        t1_glob = "t1_*.nii.gz"
        t1s = self.inimgs(t1_dir, t1_glob, src=self.OUTPUT)
        if not t1s:
            self.no_data(f"No T1 maps found in {t1_dir} matching {t1_glob}")

        LOG.info(" - T1 scale factors (left/right) : %.4f / %.4f" % (sf_left, sf_right))
        for t1 in t1s:
            left_data = t1.data * sf_left
            right_data = t1.data * sf_right
            t1.save_derived(
                left_data,
                self.outfile(t1.fname.replace(".nii.gz", "_scaled_left.nii.gz")),
            )
            t1.save_derived(
                right_data,
                self.outfile(t1.fname.replace(".nii.gz", "_scaled_right.nii.gz")),
            )


MODULES = [
    misc.ScanDates("scan_dates", input={
        "../fsort/t1w" : "*.nii.gz",
        "../fsort/t2w" : "*.nii.gz",
        "../fsort/t1_molli" : "*.nii.gz",
    }),
    maps.DixonClassify(dixon_src="../fsort/raw_dixon"),
    
    # Parameter maps
    maps.T1Molli(
        name="t1_molli_mdr",
        molli_dir="../fsort/t1_molli_raw",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=True,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1Molli(
        name="t1_molli_nomdr",
        molli_dir="../fsort/t1_molli_raw",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=False,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1Molli(
        name="t1_molli_cor",
        molli_dir="../fsort/molli_cor",
        use_raw_data=False
    ),
    maps.T1Molli(
        name="t1_molli_ax",
        molli_dir="../fsort/molli_ax",
        use_raw_data=False
    ),
    maps.T2(),
    maps.T2star(),
    maps.B0(),
    maps.B1(),
    maps.FatFractionDixon(name="ff_dixon", dixon_dir="../fproc/dixon_classify"),
    maps.AslMoco(name="pcasl_moco", asl_glob="pcasl*.nii.gz"),
    maps.AslMoco(name="fair_moco", asl_glob="fair*.nii.gz"),
    maps.T2starDixon(name="t2star_dixon", dixon_dir="../fproc/dixon_classify"),

    # Segmentations
    segmentations.KidneyT1(
        map_dir="t1_molli_cor",
        map_glob="t1_conf.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT1(
        name="seg_kidney_t1_mdr",
        map_dir="t1_molli_mdr",
        map_glob="t1_map.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT1(
        name="seg_kidney_t1_nomdr",
        map_dir="t1_molli_nomdr",
        map_glob="t1_map.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT2wRenalSegmentor(name="seg_kidney_t2w"),
    segmentations.KidneyT2w(name="seg_kidney_t2w_model2"),
    segmentations.KidneyCystT2w(
        t2w_dir="t2w", t2w_glob="t2w.nii.gz", t2w_src=Module.INPUT
    ),
    segmentations.KidneyCystTraceData(),
    segmentations.OrgansTraceData(),
    # segmentations.BodyDixon(),
    segmentations.SatDixon(name="seg_sat_dixon", dixon_dir="../fproc/dixon_classify"),
    segmentations.LiverDixon(name="seg_liver_dixon", dixon_dir="../fproc/dixon_classify"),
    segmentations.SpleenDixon(name="seg_spleen_dixon", dixon_dir="../fproc/dixon_classify"),
    segmentations.KidneyDixon(
        name="seg_kidney_dixon", dixon_dir="../fproc/dixon_classify", model_id="422"
    ),
    segmentations.PancreasEthrive(),
    segmentations.BodyDixon(name="seg_body_dixon", dixon_dir="../fproc/dixon_classify"),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive", "pancreas.nii.gz"),
    segmentations.VatDixon(
        name="seg_vat_dixon_local",
        ff_dir="ff_dixon",
        ff_glob="fat_fraction.nii.gz",
        body_dir="seg_body_dixon",
        sat_dir="seg_sat_dixon",
        organs={
            "seg_liver_dixon": "liver.nii.gz",
            "seg_spleen_dixon": "spleen.nii.gz",
            "seg_pancreas_ethrive_largestblob": "pancreas.nii.gz",
            "seg_kidney_dixon": "kidney.nii.gz",
        },
    ),
    segmentations.TotalSeg(
        name="totalseg", src_dir="../fproc/dixon_classify", dilate=1,
    ),
    segmentations.VatDixon(
        name="seg_vat_dixon_totalseg",
        ff_dir="ff_dixon",
        ff_glob="fat_fraction.nii.gz",
        body_dir="seg_body_dixon",
        sat_dir="totalseg",
        sat_glob="subcutaneous_fat.nii.gz",
        dixon_dir="../fproc/dixon_classify",
        organs={
            "totalseg": "liver.nii.gz",
            "totalseg": "spleen.nii.gz",
            "totalseg": "pancreas.nii.gz",
            "totalseg": "kidneys.nii.gz",
        },
    ),
    segmentations.TraceSeg(name="traceseg", src_dir="t2w", img_glob="t2w.nii.gz"),
    segmentations.KidneyWholeTrace(),

    # Manual fixes
    T1Scaled(),
    seg_postprocess.SegFix(
        "seg_kidney_t1",
        fix_dir_option="seg_kidney_t1_fix",
        segs={
            "*cortex_l*.nii.gz": {
                "glob": "%s/*cortex*.nii.gz",
                "side": "left",
                "fname": "kidney_cortex_l.nii.gz",
            },
            "*cortex_r*.nii.gz": {
                "glob": "%s/*cortex*.nii.gz",
                "side": "right",
                "fname": "kidney_cortex_r.nii.gz",
            },
            "*medulla_l*.nii.gz": {
                "glob": "%s/*medulla*.nii.gz",
                "side": "left",
                "fname": "kidney_medulla_l.nii.gz",
            },
            "*medulla_r*.nii.gz": {
                "glob": "%s/*medulla*.nii.gz",
                "side": "right",
                "fname": "kidney_medulla_r.nii.gz",
            },
        },
        map_dir="t1_molli_cor",
        map_fname="t1_map.nii.gz",
    ),
    seg_postprocess.SegFix(
        "seg_kidney_cyst_t2w",
        fix_dir_option="cyst_masks",
        segs={
            "kidney_cyst_mask.nii.gz": "%s/*FIX*.nii.gz",
        },
        map_dir="t2w",
        map_fname="t2w.nii.gz",
        map_src=Module.INPUT,
    ),
    seg_postprocess.SegFix(
        "seg_kidney_t2w",
        fix_dir_option="seg_kidney_t2w_fix",
        segs={
            "*mask*.nii.gz": {
                "fname": "kidney_mask.nii.gz",
                "glob": "%s.nii.gz",
            },
            "*left*.nii.gz": {
                "fname": "kidney_left.nii.gz",
                "glob": "%s.nii.gz",
                "side": "left",
            },
            "*right*.nii.gz": {
                "fname": "kidney_right.nii.gz",
                "glob": "%s.nii.gz",
                "side": "right",
            },
        },
        map_dir="t2w",
        map_fname="t2w.nii.gz",
        map_src=Module.INPUT,
    ),
    # Re-alignments
    align.FlirtAlignOnly(
        name="seg_kidney_t1_align_t2star",
        in_dir="t1_molli_cor",
        in_glob="t1_map.nii.gz",
        ref_dir="t2star",
        ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w_fix",
        weight_mask="kidney_mask.nii.gz",
        weight_mask_dil=6,
        also_align={
            "seg_kidney_t1_fix": "kidney*.nii.gz",
            "t1_molli_cor": "t1_conf.nii.gz",
        },
    ),
    # Segmentation cleaning
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean",
        srcdir="seg_kidney_t1_align_t2star",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="seg_kidney_t1_align_t2star",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean_native",
        srcdir="seg_kidney_t1_fix",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_cor",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_mdr_clean_native",
        srcdir="seg_kidney_t1_mdr",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_mdr",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean_native_generic",
        srcdir="seg_kidney_t1_fix",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_cor",
        t1_map_glob="t1_map.nii.gz",
        t2w=False,
    ),
    seg_postprocess.KidneyCystClean(
        cyst_dir="seg_kidney_cyst_t2w_fix",
        seg_t2w_dir="seg_kidney_t2w_fix",
        t2w_dir="t2w",
        t2w_glob="t2w.nii.gz",
        t2w_src=Module.INPUT,
    ),
    seg_postprocess.SegVolumes(
        "seg_kidney_t2w_vols",
        seg_dir="seg_kidney_t2w_fix",
        segs={
            "kv_left": "*left*.nii.gz",
            "kv_right": "*right*.nii.gz",
            "kv_mask": "*mask*.nii.gz",
        },
    ),
    segmentations.KidneyCortexMedullaT2w(
        t2w_seg_dir="seg_kidney_t2w_fix",
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon",
        ff_dir="ff_dixon",
        ff_glob="fat_fraction.nii.gz",
        kidney_seg_dir="seg_kidney_dixon",
        kidney_seg_glob="kidney.nii.gz",
        ff_thresh=15,
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_totalseg",
        ff_dir="ff_dixon",
        ff_glob="fat_fraction.nii.gz",
        kidney_seg_dir="totalseg",
        kidney_seg_glob="kidneys.nii.gz",
        ff_thresh=15,
    ),
    segmentations.KidneyPelvisT2w(
        t2w_seg_dir="seg_kidney_t2w_fix",
        t1_seg_dir="seg_kidney_t1_clean_native",
    ),
    segmentations.KidneyPelvisTrace(
        paren_dir="seg_kidney_t2w_fix",
        whole_dir="seg_kidney_whole_trace",
        whole_glob="kidney_whole.nii.gz",
    ),
    segmentations.OrganFat(
        name="seg_kidney_pelvis_fat_t2w",
        ff_dir="ff_dixon",
        ff_glob="fat_fraction.nii.gz",
        ff_thresh=15,
        seg_dir="seg_kidney_pelvis_t2w",
        seg_glob="kidney_pelvis*.nii.gz",
    ),
    segmentations.OrganFat(
        name="seg_kidney_pelvis_fat_trace",
        ff_dir="ff_dixon",
        ff_glob="fat_fraction.nii.gz",
        ff_thresh=15,
        seg_dir="seg_kidney_pelvis_trace",
        seg_glob="kidney_pelvis*.nii.gz",
    ),
    # Statistics and numerical measures
    statistics.Radiomics(
        name="t2w_radiomics",
        deps=["seg_kidney_t2w_fix"],
        params={
            "t2w": {"dir": "../fsort/t2w", "fname": "t2w.nii.gz"},
        },
        segs={
            "tkv_l": {"dir": "seg_kidney_t2w_fix", "fname": "*left*.nii.gz"},
            "tkv_r": {"dir": "seg_kidney_t2w_fix", "fname": "*right*.nii.gz"},
        },
        image_types=["Original", "Exponential"],
        features=["firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"],
    ),
    Stats(),
    StatsDixon(),
    StatsDixonTotalseg(),
    statistics.CMD(
        cmd_params=["t2star", "t2star", "t1"],
        skip_params=["t1_noclean"],
    ),
    T1MolliMetadata(),
    statistics.ShapeMetrics(
        name="tkv_shape_metrics",
        seg_dir="seg_kidney_t2w_fix",
        segs={"tkv_l": "*left*.nii.gz", "tkv_r": "*right*.nii.gz"},
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics",
        seg_dir="seg_kidney_dixon",
        segs={
            "kidney_left": "kidney_left.nii.gz",
            "kidney_right": "kidney_right.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_totalseg",
        seg_dir="totalseg",
        segs={
            "kidney_left": "kidney_left.nii.gz",
            "kidney_right": "kidney_right.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="spleen_shape_metrics_totalseg",
        seg_dir="seg_spleen_dixon",
        segs={
            "spleen": "spleen.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="liver_shape_metrics_totalseg",
        seg_dir="seg_liver_dixon",
        segs={
            "liver": "liver.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.ShapeMetrics(
        name="pancreas_shape_metrics_totalseg",
        seg_dir="totalseg",
        segs={
            "pancreas": "pancreas.nii.gz",
        },
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.Radiomics(
        name="tkv_radiomics",
        deps=["seg_kidney_t2w_fix"],
        params={
            "t2w": {"dir": "t2w", "fname": "t2w.nii.gz", "src": Module.INPUT},
        },
        segs={
            "tkv_l": {"dir": "seg_kidney_t2w_fix", "fname": "*left*.nii.gz"},
            "tkv_r": {"dir": "seg_kidney_t2w_fix", "fname": "*right*.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="kidney_dixon_radiomics",
        deps=["t1_molli_cor", "seg_kidney_dixon"],
        params={
            "t1": {
                "dir": "t1_molli_cor",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_l": {
                "dir": "seg_kidney_dixon",
                "fname": "*left*.nii.gz",
            },
            "kidney_dixon_r": {
                "dir": "seg_kidney_dixon",
                "fname": "*right*.nii.gz",
            },
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="kidney_dixon_radiomics_totalseg",
        deps=["t1_molli_cor", "totalseg"],
        params={
            "t1": {
                "dir": "t1_molli_cor",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_l": {
                "dir": "totalseg",
                "fname": "*kidney_left*.nii.gz",
            },
            "kidney_dixon_r": {
                "dir": "totalseg",
                "fname": "*kidney_right*.nii.gz",
            },
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="organ_dixon_radiomics",
        deps=["t1_molli_cor", "totalseg"],
        params={
            "t1": {
                "dir": "t1_molli_cor",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_l": {
                "dir": "totalseg",
                "fname": "kidney_left.nii.gz",
            },
            "kidney_dixon_r": {
                "dir": "totalseg",
                "fname": "kidney_right.nii.gz",
            },
            "liver" : {
                "dir": "totalseg",
                "fname" : "liver.nii.gz",
            },
            "pancreas" : {
                "dir": "totalseg",
                "fname" : "pancreas.nii.gz",
            },
            "spleen" : {
                "dir": "totalseg",
                "fname" : "spleen.nii.gz",
            },
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="t1_radiomics_left",
        deps=["t1_scaled", "t2", "seg_kidney_t1_clean_native"],
        params={
            "t1": {"dir": "t1_scaled", "fname": "t1_conf_scaled_left.nii.gz"},
        },
        segs={
            "cortex_l": {
                "dir": "seg_kidney_t1_clean_native",
                "fname": "*cortex_l*.nii.gz",
            },
            "medulla_l": {
                "dir": "seg_kidney_t1_clean_native",
                "fname": "*medulla_l*.nii.gz",
            },
        },
    ),
    statistics.Radiomics(
        name="t1_radiomics_right",
        deps=["t1_scaled", "t2", "seg_kidney_t1_clean_native"],
        params={
            "t1": {"dir": "t1_scaled", "fname": "t1_conf_scaled_right.nii.gz"},
        },
        segs={
            "cortex_r": {
                "dir": "seg_kidney_t1_clean_native",
                "fname": "*cortex_r*.nii.gz",
            },
            "medulla_r": {
                "dir": "seg_kidney_t1_clean_native",
                "fname": "*medulla_r*.nii.gz",
            },
        },
    ),
    statistics.ShapeMetrics(
        name="wkv_shape_metrics",
        seg_dir="seg_kidney_whole",
        segs={"wkv_l": "*kidney_whole_l.nii.gz", "wkv_r": "*kidney_whole_r.nii.gz"},
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.Radiomics(
        name="wkv_radiomics",
        deps=["t2w", "seg_kidney_whole"],
        params={
            "t2w": {"dir": "t2w", "fname": "t2w.nii.gz", "src": Module.INPUT},
        },
        segs={
            "wkv_l": {"dir": "seg_kidney_whole", "fname": "*kidney_whole_l.nii.gz"},
            "wkv_r": {"dir": "seg_kidney_whole", "fname": "*kidney_whole_r.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ), 
    statistics.ShapeMetrics(
        name="wkv_ero_shape_metrics",
        seg_dir="seg_kidney_pelvis_trace",
        segs={"wkv_ero_l": "*ero_l*.nii.gz", "wkv_ero_r": "*ero_r*.nii.gz"},
        metrics=[
            "surf_area",
            "surf_area_over_vol",
            "vol",
            "compactness",
            "long_axis",
            "short_axis",
            "mi1",
            "mi2",
            "mi3",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.Radiomics(
        name="wkv_ero_radiomics",
        deps=["t2w", "seg_kidney_pelvis_trace_splitlr"],
        params={
            "t2w": {"dir": "t2w", "fname": "t2w.nii.gz", "src": Module.INPUT},
        },
        segs={
            "wkv_ero_l": {"dir": "seg_kidney_pelvis_trace_splitlr", "fname": "*ero_l*.nii.gz"},
            "wkv_ero_r": {"dir": "seg_kidney_pelvis_trace_splitlr", "fname": "*ero_r*.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    
    statistics.KidneyCystStats(
        name="kidney_cyst_stats",
        cyst_dir="seg_kidney_cyst_t2w_clean",
        cyst_glob="kidney_cyst_mask.nii.gz",
        suffix="uon",
    ),
    statistics.KidneyCystStats(
        name="kidney_cyst_stats_trace_fixed",
        cyst_dir="seg_kidney_cyst_trace",
        cyst_glob="kidney_cyst_fixed.nii.gz",
        suffix="defin",
    ),
    statistics.KidneyCystStats(
        name="kidney_cyst_stats_trace_orig",
        cyst_dir="seg_kidney_cyst_trace",
        cyst_glob="kidney_cyst_orig.nii.gz",
        suffix="trace",
    ),
    statistics.SegStats(
        name="wkv_volumes",
        segs={
            "wkv_all": {
                "dir": "seg_kidney_whole",
                "glob": "kidney_whole.nii.gz",
            },
            "wkv_l": {
                "dir": "seg_kidney_whole",
                "glob": "kidney_whole_l.nii.gz",
            },
            "wkv_r": {
                "dir": "seg_kidney_whole",
                "glob": "kidney_whole_r.nii.gz",
            },
        },
        seg_volumes=True,
    ),
    statistics.SegStats(
        "wkv_ero_volumes",
        segs={
            "wkv_ero_all" : {
                "dir" : "seg_kidney_pelvis_trace",
                "glob" : "whole_kidney_ero.nii.gz",
            },
            "wkv_ero_all_l" : {
                "dir" : "seg_kidney_pelvis_trace_splitlr",
                "glob" : "whole_kidney_ero_l.nii.gz",
            },
            "wkv_ero_all_r" : {
                "dir" : "seg_kidney_pelvis_trace_splitlr",
                "glob" : "whole_kidney_ero_r.nii.gz",
            },
        },
        seg_volumes=True,
    ),
    statistics.SegStats(
        name="kidney_pelvis_stats_trace",
        segs={
            "kidney_pelvis_l_trace": {
                "dir": "seg_kidney_pelvis_trace",
                "glob": "kidney_pelvis_left.nii.gz",
                "params": [],
            },
            "kidney_pelvis_r_trace": {
                "dir": "seg_kidney_pelvis_trace",
                "glob": "kidney_pelvis_right.nii.gz",
                "params": [],
            },
            "kidney_pelvis_fat_l_trace" : {
                "dir": "seg_kidney_pelvis_fat_trace",
                "glob": "kidney_pelvis_left_fat.nii.gz",
                "params" : ["ff"],
            },
            "kidney_pelvis_fat_r_trace" : {
                "dir": "seg_kidney_pelvis_fat_trace",
                "glob": "kidney_pelvis_right_fat.nii.gz",
                "params" : ["ff"],
            },
            "kidney_pelvis_nofat_l_trace" : {
                "dir": "seg_kidney_pelvis_fat_trace",
                "glob": "kidney_pelvis_left_nofat.nii.gz",
                "params": [],
            },
            "kidney_pelvis_nofat_r_trace" : {
                "dir": "seg_kidney_pelvis_fat_trace",
                "glob": "kidney_pelvis_right_nofat.nii.gz",
                "params": [],
            },
            "kidney_paren_l" : {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_left.nii.gz",
                "params" : ["ff", "ff_lt25"],
            },
            "kidney_paren_r" : {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_right.nii.gz",
                "params" : ["ff", "ff_lt25"],
            },
            "kidney_cyst_l" : {
                "dir": "seg_kidney_cyst_trace_splitlr",
                "glob": "*_l.nii.gz",
                "params" : [],
            },
            "kidney_cyst_r" : {
                "dir": "seg_kidney_cyst_trace_splitlr",
                "glob": "*_r.nii.gz",
                "params" : [],
            },
        },
        params={
            "ff" : {
                "dir": "ff_dixon",
                "glob": "fat_fraction.nii.gz",
            },
            "ff_lt25" : {
                "dir": "ff_dixon",
                "glob": "fat_fraction.nii.gz",
                "limits": (0, 25),
            }
        },
        seg_volumes=True,
        stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd", "perc90", "te", "mode", "fwhm"],
    ),
    statistics.SegStats(
        name="kidney_stats_alternate",
        segs={
            "kidney_paren_defin" : {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_mask.nii.gz",
                "params": [],
            },
            "kidney_paren_model1" : {
                "dir" : "seg_kidney_t2w_model1",
                "glob" : "kidney_mask.nii.gz",
                "params": [],
            },
            "kidney_paren_model2" : {
                "dir" : "seg_kidney_t2w",
                "glob" : "kidney_mask.nii.gz",
                "params": [],
            },
            "kidney_pelvis_defin": {
                "dir": "seg_kidney_pelvis_trace",
                "glob": "kidney_pelvis.nii.gz",
                "params": [],
            },
            "kidney_pelvis_t2w" : {
                "dir": "seg_kidney_pelvis_t2w",
                "glob": "kidney_pelvis.nii.gz",
                "params": [],
            },
        },
        seg_volumes=True,
    ),
    statistics.SegVolumeDiffs(
        name="cyst_volume_diffs",
        stats_files=[
            "kidney_cyst_stats/kidney_cyst.csv",
            "kidney_cyst_stats_trace_fixed/kidney_cyst.csv",
            "kidney_cyst_stats_trace_orig/kidney_cyst.csv",
        ],
        diffs={
            "cyst_def_min_trace" : ("kidney_cyst_defin_vol", "kidney_cyst_trace_vol"),
            "cyst_def_min_uon" : ("kidney_cyst_defin_vol", "kidney_cyst_uon_vol"),
            "cyst_trace_min_uon" : ("kidney_cyst_trace_vol", "kidney_cyst_uon_vol"),
        }
    ),
    statistics.SegStats(
        name="organ_volumes",
        segs={
            "liver_local": {
                "dir": "seg_liver_dixon",
                "glob": "kidney_whole.nii.gz",
            },
            "liver_totalseg": {
                "dir": "totalseg",
                "glob": "liver.nii.gz",
            },
            "pancreas_local": {
                "dir": "seg_pancreas_ethrive",
                "glob": "pancreas.nii.gz",
            },
            "pancreas_totalseg": {
                "dir": "totalseg",
                "glob": "pancreas.nii.gz",
            },
            "spleen_local": {
                "dir": "seg_spleen_dixon",
                "glob": "spleen.nii.gz",
            },
            "spleen_totalseg": {
                "dir": "totalseg",
                "glob": "spleen.nii.gz",
            },
            "sat_local": {
                "dir": "seg_sat_dixon",
                "glob": "sat.nii.gz",
            },
            "sat_totalseg": {
                "dir": "totalseg",
                "glob": "subcutaneous_fat.nii.gz",
            },
            "vat_local": {
                "dir": "seg_vat_dixon",
                "glob": "vat.nii.gz",
            },
            "vat_totalseg": {
                "dir": "seg_vat_dixon_totalseg",
                "glob": "vat.nii.gz",
            },
        },
        seg_volumes=True,
    ),
    #TempAddPelvis(),
]


def add_options(parser):
    parser.add_argument(
        "--t1-scale-factors", help="Directory containing manual fixed cyst masks"
    )
    parser.add_argument(
        "--cyst-masks", help="Directory containing manual fixed cyst masks"
    )
    parser.add_argument(
        "--seg-kidney-t2w-fix", help="Directory containing manual T2w kidney masks"
    )
    parser.add_argument(
        "--seg-kidney-t1-fix",
        help="Directory containing manual kidney cortex/medulla masks + maps",
    )
    parser.add_argument(
        "--seg-kidney-cyst-trace", help="Directory containing TRACE cyst segmentations"
    )
    parser.add_argument(
        "--seg-organs-trace", help="Directory containing TRACE organ segmentations"
    )
