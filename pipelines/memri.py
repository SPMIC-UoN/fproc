import logging
import os

import numpy as np

from fproc.module import Module
from fproc.modules import (
    maps,
    segmentations,
    statistics,
    seg_postprocess,
    regrid,
    misc,
)

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME = "memri"

class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats",
            default_limits="3t",
            segs={
                "kidney_cortex": {
                    "dir": "seg_kidney_t1_clean_fix",
                    "glob": "*cortex*.nii.gz",
                },
                "kidney_cortex_l": {
                    "dir": "seg_kidney_t1_clean_fix",
                    "glob": "*cortex_l*.nii.gz",
                },
                "kidney_cortex_r": {
                    "dir": "seg_kidney_t1_clean_fix",
                    "glob": "*cortex_r*.nii.gz",
                },
                "kidney_medulla": {
                    "dir": "seg_kidney_t1_clean_fix",
                    "glob": "*medulla*.nii.gz",
                },
                "kidney_medulla_l": {
                    "dir": "seg_kidney_t1_clean_fix",
                    "glob": "*medulla_l*.nii.gz",
                },
                "kidney_medulla_r": {
                    "dir": "seg_kidney_t1_clean_fix",
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
                "t1": {
                    "dir": "t1_molli_stitch",
                    "glob": "t1_conf.nii.gz",
                },
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd", "perc90", "te", "mode", "fwhm"],
        )

class T1MolliMetadata(Module):
    def __init__(self, name="t1_molli_md", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t1_dir = "t1_molli"
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

MODULES = [
    misc.ScanDates("scan_dates", input={
        "../fsort/t2w" : "*.nii.gz",
        "../fsort/t1_molli" : "*.nii.gz",
    }),
    # Parameter maps
    maps.T1Molli(
        name="t1_molli",
        molli_dir="../fsort/t1_molli",
        molli_glob="t1_molli_raw*.nii.gz",
        t1_thresh=(0, 5000),
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    # Stitch together potentially multiple slice maps
    regrid.StitchSlices(
        name="t1_molli_stitch",
        img_dir="t1_molli",
        imgs={
            "*t1_map*.nii.g?": "t1_map.nii.gz",  # Just to make sure globs are unique keys!
            "*t1_map*.nii.gz": "t1_conf.nii.gz",
            "*t1_conf*.nii.gz": "t1_conf.nii.gz",
        },
    ),
    # Segmentations
    segmentations.KidneyT1(
        map_dir="t1_molli_stitch",
        map_glob="t1_conf.nii.gz",
        t1_limits=[(0, 0), (4136, 0)],
    ),
    segmentations.KidneyT2wRenalSegmentor(
        name="seg_kidney_t2w"
    ),
    seg_postprocess.SegFix(
        "seg_kidney_t2w",
        fix_dir_option="seg_kidney_t2w_fix",
        segs={
            "*left*.nii.gz": {
                "glob": "%s/*left*.nii.gz",
                "fname": "kidney_left.nii.gz",
            },
            "*right*.nii.gz": {
                "glob": "%s/*right*.nii.gz",
                "fname": "kidney_right.nii.gz",
            },
            "*kidney_mask*.nii.gz": {
                "glob": "%s/*kidney_mask*.nii.gz",
                "fname": "kidney_mask.nii.gz",
            },
        }
    ),
    
    # Re-alignments
    # Segmentation cleaning
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean",
        srcdir="seg_kidney_t1",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_stitch",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.SegFix(
        "seg_kidney_t1_clean",
        fix_dir_option="seg_kidney_t1_fix",
        segs={
            "*cortex_l*.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*cortex_l_t1fix*.nii.gz",
                "fname": "kidney_cortex_l_t1.nii.gz",
            },
            "*cortex_r*.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*cortex_r_t1fix*.nii.gz",
                "fname": "kidney_cortex_r_t1.nii.gz",
            },
            "*cortex_t1.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*cortex_t1fix*.nii.gz",
                "fname": "kidney_cortex_t1.nii.gz",
            },
            "*medulla_l*.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*medulla_l_t1fix*.nii.gz",
                "fname": "kidney_medulla_l_t1.nii.gz",
            },
            "*medulla_r*.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*medulla_r_t1fix*.nii.gz",
                "fname": "kidney_medulla_r_t1.nii.gz",
            },
            "*medulla_t1.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*medulla_t1fix*.nii.gz",
                "fname": "kidney_medulla_t1.nii.gz",
            },
            "*_all_l*.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*_all_l_t1fix*.nii.gz",
                "fname": "kidney_all_l_t1.nii.gz",
            },
            "*_all_r*.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*_all_r_t1fix*.nii.gz",
                "fname": "kidney_all_r_t1.nii.gz",
            },
             "*_all_t1.nii.gz": {
                "glob": "%s/seg_kidney_t1_clean/*_all_t1fix*.nii.gz",
                "fname": "kidney_all_t1.nii.gz",
            },
        },
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
    segmentations.KidneyPelvisT2w(
        t2w_seg_dir="seg_kidney_t2w_fix",
        t1_seg_dir="seg_kidney_t1_clean_fix",
    ),
    # Statistics and numerical measures
    statistics.Radiomics(
        name="t2w_radiomics",
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
    statistics.CMD(
        cmd_params=["t1"],
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
    statistics.Radiomics(
        name="tkv_radiomics",
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
        name="t1_radiomics_left",
        params={
            "t1": {"dir": "t1_molli", "fname": "t1_conf.nii.gz"},
        },
        segs={
            "cortex_l": {
                "dir": "seg_kidney_t1_clean_fix",
                "fname": "*cortex_l*.nii.gz",
            },
            "medulla_l": {
                "dir": "seg_kidney_t1_clean_fix",
                "fname": "*medulla_l*.nii.gz",
            },
        },
    ),
    statistics.Radiomics(
        name="t1_radiomics_right",
        params={
            "t1": {"dir": "t1_molli", "fname": "t1_conf.nii.gz"},
        },
        segs={
            "cortex_r": {
                "dir": "seg_kidney_t1_clean_fix",
                "fname": "*cortex_r*.nii.gz",
            },
            "medulla_r": {
                "dir": "seg_kidney_t1_clean_fix",
                "fname": "*medulla_r*.nii.gz",
            },
        },
    ),
    statistics.SegStats(
        name="kidney_stats",
        segs={
            "kidney_paren" : {
                "dir": "seg_kidney_t2w_fix",
                "glob": "kidney_mask.nii.gz",
                "params": [],
            },
            "kidney_pelvis": {
                "dir": "seg_kidney_pelvis_t2w",
                "glob": "kidney_pelvis.nii.gz",
                "params": [],
            },
        },
        seg_volumes=True,
    ),
]

def add_options(parser):
    parser.add_argument(
        "--seg-kidney-t2w-fix", help="Directory containing manual fixed t2w kidney masks"
    )
    parser.add_argument(
        "--seg-kidney-t1-fix", help="Directory containing manual fixed t1 kidney masks"
    )