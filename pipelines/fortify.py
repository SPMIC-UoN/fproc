# FORTIFY pipeline

import logging

import numpy as np

from fproc.module import Module
from fproc.modules import (
    maps,
    segmentations,
    statistics,
    seg_postprocess,
    align,
    misc,
)

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME = "afirm"


class KidneyStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="kidney_stats",
            default_limits="3t",
            seg_volumes=True,
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
                "t2_stim": {
                    "dir": "t2",
                    "glob": "t2_stim.nii.gz",
                    "segs": [
                        "kidney_cortex_l",
                        "kidney_cortex_r",
                        "kidney_medulla_l",
                        "kidney_medulla_r",
                    ],
                },
                "b1_stim": {
                    "dir": "t2",
                    "glob": "b1_stim.nii.gz",
                    "segs": [
                        "kidney_cortex_l",
                        "kidney_cortex_r",
                        "kidney_medulla_l",
                        "kidney_medulla_r",
                        "kidney_l",
                        "kidney_r",
                    ],
                },
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
                    "dir": "t1_molli",
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
                "t1_mdr": {
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
                "adc_mdr": {
                    "dir": "dwi_adc",
                    "glob": "*adc_map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_se_clean_native"},
                    },
                },
            },
            stats=[
                "n",
                "vol",
                "iqn",
                "iqvol",
                "iqmean",
                "median",
                "iqstd",
                "perc90",
                "te",
                "mode",
                "fwhm",
            ],
        )


class KidneyStatsRpt(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="kidney_stats_rpt",
            default_limits="3t",
            seg_volumes=True,
            segs={
                "kidney_cortex_rpt": {
                    "dir": "seg_kidney_t1_rpt_clean",
                    "glob": "*cortex*.nii.gz",
                },
                "kidney_cortex_l_rpt": {
                    "dir": "seg_kidney_t1_rpt_clean",
                    "glob": "*cortex_l*.nii.gz",
                },
                "kidney_cortex_r_rpt": {
                    "dir": "seg_kidney_t1_rpt_clean",
                    "glob": "*cortex_r*.nii.gz",
                },
                "kidney_medulla_rpt": {
                    "dir": "seg_kidney_t1_rpt_clean",
                    "glob": "*medulla*.nii.gz",
                },
                "kidney_medulla_l_rpt": {
                    "dir": "seg_kidney_t1_rpt_clean",
                    "glob": "*medulla_l*.nii.gz",
                },
                "kidney_medulla_r_rpt": {
                    "dir": "seg_kidney_t1_rpt_clean",
                    "glob": "*medulla_r*.nii.gz",
                },
                "tkv_l_rpt": {
                    "dir": "seg_kidney_t2w_fix",
                    "glob": "*left*.nii.gz",
                },
                "tkv_r_rpt": {
                    "dir": "seg_kidney_t2w_fix",
                    "glob": "*right*.nii.gz",
                },
            },
            params={
                "t2_stim": {
                    "dir": "t2_rpt",
                    "glob": "t2_stim.nii.gz",
                    "segs": [
                        "kidney_cortex_l",
                        "kidney_cortex_r",
                        "kidney_medulla_l",
                        "kidney_medulla_r",
                    ],
                },
                "b1_stim": {
                    "dir": "t2_rpt",
                    "glob": "b1_stim.nii.gz",
                    "segs": [
                        "kidney_cortex_l",
                        "kidney_cortex_r",
                        "kidney_medulla_l",
                        "kidney_medulla_r",
                        "kidney_l",
                        "kidney_r",
                    ],
                },
                "t2star_exp": {
                    "dir": "t2star_rpt",
                    "glob": "t2star_2p_exp*.nii.gz",
                },
                "t2star_loglin": {
                    "dir": "t2star_rpt",
                    "glob": "t2star_loglin*.nii.gz",
                },
                "r2star_exp": {
                    "dir": "t2star_rpt",
                    "glob": "r2star_2p_exp*.nii.gz",
                },
                "r2star_loglin": {
                    "dir": "t2star_rpt",
                    "glob": "r2star_loglin*.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_rpt",
                    "glob": "t1_conf.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {"dir": "seg_kidney_t1_rpt_clean_native"},
                        "kidney_cortex_r": {"dir": "seg_kidney_t1_rpt_clean_native"},
                        "kidney_cortex": {"dir": "seg_kidney_t1_rpt_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_rpt_clean_native"},
                        "kidney_medulla_l": {"dir": "seg_kidney_t1_rpt_clean_native"},
                        "kidney_medulla_r": {"dir": "seg_kidney_t1_rpt_clean_native"},
                    },
                },
                "t1_mdr": {
                    "dir": "t1_molli_rpt_mdr",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex_l": {
                            "dir": "seg_kidney_t1_rpt_mdr_clean_native"
                        },
                        "kidney_cortex_r": {
                            "dir": "seg_kidney_t1_rpt_mdr_clean_native"
                        },
                        "kidney_cortex": {"dir": "seg_kidney_t1_rpt_mdr_clean_native"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_rpt_mdr_clean_native"},
                        "kidney_medulla_l": {
                            "dir": "seg_kidney_t1_rpt_mdr_clean_native"
                        },
                        "kidney_medulla_r": {
                            "dir": "seg_kidney_t1_rpt_mdr_clean_native"
                        },
                    },
                },
            },
            stats=[
                "n",
                "vol",
                "iqn",
                "iqvol",
                "iqmean",
                "median",
                "iqstd",
                "perc90",
                "te",
                "mode",
                "fwhm",
            ],
        )


class PadT1SEData(Module):
    """
    Pad out 7 volume T1 data to a 20 volume image so it can go through the
    T1_SE segmentor
    """

    def __init__(self, name="t1_se_pad", **kwargs):
        self._t1_se_dir = kwargs.get("t1_se_dir", "t1_se")
        deps = [self._t1_se_dir]
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        t1_se_glob = self.kwargs.get("t1_se_glob", "t1_se*.nii.gz")

        t1_se = self.single_inimg(self._t1_se_dir, t1_se_glob, src=self.OUTPUT)
        if not t1_se:
            self.no_data(f"No T1 SE data found in {self._t1_se_dir}/{t1_se_glob}")
        data = t1_se.data
        if data.ndim != 4 or data.shape[3] != 7:
            self.no_data(
                f"Unexpected T1 SE data shape: {data.shape} - expected 4D with 7 volumes"
            )

        padded_data = np.zeros(list(data.shape[:3]) + [20], dtype=data.dtype)
        for src_vol, dest_vol in enumerate([4, 5, 6, 9, 11, 13, 15]):
            padded_data[..., dest_vol] = data[..., src_vol]
        t1_se.save_derived(padded_data, self.outfile(t1_se.fname))


MODULES = [
    misc.ScanDates(
        "scan_dates",
        input={
            "../fsort/t1w": "*.nii.gz",
            "../fsort/t2w": "*.nii.gz",
            "../fsort/t1_molli": "*.nii.gz",
        },
    ),
    # Parameter maps
    maps.T1Molli(
        name="t1_molli",
        molli_dir="../fsort/t1_molli",
        t1_thresh=(0, 5000),
        use_raw_data=False,
    ),
    maps.T1Molli(
        name="t1_molli_mdr",
        molli_dir="../fsort/t1_molli_raw",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=True,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    # As above for repeat scans
    maps.T1Molli(
        name="t1_molli_rpt",
        molli_dir="../fsort/t1_molli_rpt",
        t1_thresh=(0, 5000),
        use_raw_data=False,
    ),
    maps.T1Molli(
        name="t1_molli_rpt_mdr",
        molli_dir="../fsort/t1_molli_rpt_raw",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=True,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T2(),
    maps.T2(name="t2_rpt", t2_dir="t2_rpt"),
    maps.T2star(),
    maps.T2star(name="t2star_rpt", t2star_dir="t2star_rpt"),
    maps.B0(),
    maps.FatFractionDixon(dixon_dir="../fsort/dixon"),
    maps.DwiMoco(),
    maps.DwiAdc(),
    # Segmentations
    segmentations.KidneyT1(
        name="seg_kidney_t1",
        map_dir="t1_molli",
        map_glob="t1_conf.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT1(
        name="seg_kidney_t1_mdr",
        map_dir="t1_molli_mdr",
        map_glob="t1_map.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    # As above for repeat scans
    segmentations.KidneyT1(
        name="seg_kidney_t1_rpt",
        map_dir="t1_molli_rpt",
        map_glob="t1_conf.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT1(
        name="seg_kidney_t1_rpt_mdr",
        map_dir="t1_molli_rpt_mdr",
        map_glob="t1_map.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    PadT1SEData(
        t1_se_dir="../fsort/t1_se_raw",
        t1_se_glob="t1_se_mag*.nii.gz",
    ),
    segmentations.KidneyT1SE(
        name="seg_kidney_t1_se",
        t1_se_dir="t1_se_pad",
        t1_se_glob="t1_se_mag.nii.gz",
        t1_ref_glob="t1_se_mag.nii.gz",
    ),
    segmentations.KidneyT2wRenalSegmentor(name="seg_kidney_t2w"),
    segmentations.KidneyCystTraceData(
        trace_cyst_data="/spmstore/project/RenalMRI/fortify/TRACE_CYST_MASKS"
    ),
    segmentations.OrgansTraceData(
        trace_organs_data="/spmstore/project/RenalMRI/fortify/TRACE_ORGAN_MASKS"
    ),
    seg_postprocess.SegFix(
        "seg_kidney_t2w",
        fix_dir_option="seg_kidney_t2w_fix",
        segs={
            "*mask*.nii.gz": {
                "fname": "kidney_mask.nii.gz",
                "glob": "%s/kidney_mask_FIX.nii.gz",
            },
            "*left*.nii.gz": {
                "fname": "kidney_left.nii.gz",
                "glob": "%s/kidney_mask_FIX.nii.gz",
                "side": "left",
            },
            "*right*.nii.gz": {
                "fname": "kidney_right.nii.gz",
                "glob": "%s/kidney_mask_FIX.nii.gz",
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
        in_dir="t1_molli",
        in_glob="t1_map.nii.gz",
        ref_dir="t2star",
        ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w_fix",
        weight_mask="kidney_mask.nii.gz",
        weight_mask_dil=6,
        also_align={
            "seg_kidney_t1": "kidney*.nii.gz",
            "t1_molli": "t1_conf.nii.gz",
        },
    ),
    align.FlirtAlignOnly(
        name="seg_kidney_t1_rpt_align_t2star",
        in_dir="t1_molli_rpt",
        in_glob="t1_map.nii.gz",
        ref_dir="t2star_rpt",
        ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w_fix",
        weight_mask="kidney_mask.nii.gz",
        weight_mask_dil=6,
        also_align={
            "seg_kidney_t1_rpt": "kidney*.nii.gz",
            "t1_molli_rpt": "t1_conf.nii.gz",
        },
    ),
    ## Segmentation Post processing
    # Kidney cleaning - basic and repeats
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
        srcdir="seg_kidney_t1",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli",
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
        name="seg_kidney_t1_rpt_clean",
        srcdir="seg_kidney_t1_rpt_align_t2star",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="seg_kidney_t1_rpt_align_t2star",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_rpt_clean_native",
        srcdir="seg_kidney_t1_rpt",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_rpt",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_rpt_mdr_clean_native",
        srcdir="seg_kidney_t1_rpt_mdr",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_rpt_mdr",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    # T1 SE kidney segmentation cleaning
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_se_clean_native",
        srcdir="seg_kidney_t1_se",
        seg_t1_glob="kidney*.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    # Kidney pelvis segmentation by removing cortex/medulla (and cysts) from whole kidney
    segmentations.KidneyPelvisTrace(
        paren_dir="seg_kidney_t2w_fix",
        whole_dir="seg_kidney_trace",
        whole_glob="kidney.nii.gz",
        cyst_dir="seg_kidney_cyst_trace",
    ),
    ## Statistics and numerical measures
    KidneyStats(),
    KidneyStatsRpt(),
    statistics.KidneyCystStats(
        name="kidney_cyst_stats_trace",
        cyst_dir="seg_kidney_cyst_trace",
        cyst_glob="kidney_cyst_orig.nii.gz",
        suffix="trace",
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
    statistics.SegStats(
        name="tkv_volumes",
        segs={
            "tkv_all": {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_mask.nii.gz",
            },
            "tkv_l": {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_left.nii.gz",
            },
            "tkv_r": {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_right.nii.gz",
            },
        },
        seg_volumes=True,
    ),
    statistics.Radiomics(
        name="wkv_radiomics",
        deps=["t2w", "seg_organs_trace"],
        params={
            "t2w": {"dir": "t2w", "fname": "t2w.nii.gz", "src": Module.INPUT},
        },
        segs={
            "wkv_l": {"dir": "seg_organs_trace", "fname": "*kidney_left.nii.gz"},
            "wkv_r": {"dir": "seg_organs_trace", "fname": "*kidney_right.nii.gz"},
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
    statistics.SegStats(
        name="wkv_volumes",
        segs={
            "wkv_all": {
                "dir": "seg_organs_trace",
                "glob": "kidney_all.nii.gz",
            },
            "wkv_l": {
                "dir": "seg_organs_trace",
                "glob": "kidney_left.nii.gz",
            },
            "wkv_r": {
                "dir": "seg_organs_trace",
                "glob": "kidney_right.nii.gz",
            },
        },
        seg_volumes=True,
    ),
]


def add_options(parser):
    parser.add_argument(
        "--seg-kidney-t2w-fix", help="Directory containing manual T2w kidney masks"
    )
