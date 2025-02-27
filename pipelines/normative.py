import logging

from fproc.module import Module
from fproc.modules import maps, segmentations, regrid, seg_postprocess, align, statistics

import numpy as np

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class T1MolliMetadata(Module):
    def __init__(self, name="t1_molli_md", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t1_dir = "t1"
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

class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats", 
            default_limits="3t",
            segs={
                "kidney_cortex" : {
                    "dir" : "seg_t1_clean",
                    "glob" : "*cortex*.nii.gz"
                },
                "kidney_cortex_l" : {
                    "dir" : "seg_t1_clean",
                    "glob" : "*cortex_l*.nii.gz"
                },
                "kidney_cortex_r" : {
                    "dir" : "seg_t1_clean",
                    "glob" : "*cortex_r*.nii.gz"
                },
                "kidney_medulla" : {
                    "dir" : "seg_t1_clean",
                    "glob" : "*medulla*.nii.gz"
                },
                "kidney_medulla_l" : {
                    "dir" : "seg_t1_clean",
                    "glob" : "*medulla_l*.nii.gz"
                },
                "kidney_medulla_r" : {
                    "dir" : "seg_t1_clean",
                    "glob" : "*medulla_r*.nii.gz"
                },
                "kidney_l" : {
                    "dir" : "seg_t1_clean",
                    "glob" : "*_l.nii.gz",
                    "params" : ["b1_stim"]
                },
                "kidney_r" : {
                    "dir" : "seg_t1_clean",
                    "glob" : "*_r.nii.gz",
                    "params" : ["b1_stim"]
                },
                "tkv_l" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "*left*.nii.gz"
                },
                "tkv_r" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "*right*.nii.gz"
                },
            },
            params={
                "t2star_exp" : {
                    "dir" : "t2star",
                    "glob" : "*t2star_2p_exp*.nii.gz",
                },
                "t2star_loglin" : {
                    "dir" : "t2star",
                    "glob" : "*t2star_loglin*.nii.gz",
                },
                "r2star_exp" : {
                    "dir" : "t2star",
                    "glob" : "*r2star_2p_exp*.nii.gz",
                },
                "r2star_loglin" : {
                    "dir" : "t2star",
                    "glob" : "*r2star_loglin*.nii.gz",
                },
                "t1" : {
                    "dir" : "t1_stitched_fix",
                    "glob" : "t1_conf.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_t1_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_t1_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_t1_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_t1_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_t1_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_t1_clean_native"},
                    }
                },
                "t1_noclean" : {
                    "dir" : "t1_stitched_fix",
                    "glob" : "t1_conf.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_t1_clean_native_generic"},
                        "kidney_cortex_r" : {"dir" : "seg_t1_clean_native_generic"},
                        "kidney_cortex" : {"dir" : "seg_t1_clean_native_generic"},
                        "kidney_medulla" : {"dir" : "seg_t1_clean_native_generic"},
                        "kidney_medulla_l" : {"dir" : "seg_t1_clean_native_generic"},
                        "kidney_medulla_r" : {"dir" : "seg_t1_clean_native_generic"},
                    }
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
                "b1_rescaled" : {
                    "dir" : "b1",
                    "glob" : "b1_rescaled.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"],
                },
                "t2_exp" : {
                    "dir" : "t2",
                    "glob" : "t2_exp.nii.gz",
                    "segs" : ["kidney_cortex_l", "kidney_cortex_r", "kidney_medulla_l", "kidney_medulla_r"],
                },
                "t2_stim" : {
                    "dir" : "t2",
                    "glob" : "t2_stim.nii.gz",
                    "segs" : ["kidney_cortex_l", "kidney_cortex_r", "kidney_medulla_l", "kidney_medulla_r"],
                },
                "b1_stim" : {
                    "dir" : "t2",
                    "glob" : "b1_stim.nii.gz",
                    "segs" : ["kidney_cortex_l", "kidney_cortex_r", "kidney_medulla_l", "kidney_medulla_r", "kidney_l", "kidney_r"],
                }
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
            seg_volumes=False,
        )

NAME="normative"

MODULES = [
    maps.T1Molli(name="t1_map", molli_dir="t1", molli_glob="t1_raw_molli*.nii.gz"),
    regrid.StitchSlices(
        name="t1_stitched",
        img_dir="t1_map",
        imgs={
            "*t1_map*.nii.gz" : "t1_map.nii.gz",
            "*t1_conf*.nii.gz" : "t1_conf.nii.gz",
        }
    ),
    maps.T2(),
    maps.MTR(),
    maps.B0(),
    maps.B1(),
    maps.T2star(),
    T1MolliMetadata(),
    segmentations.KidneyT1(map_dir="t1_stitched"),
    segmentations.KidneyT2w(),
    maps.MapFix(
        "t1_stitched",
        fix_dir_option="seg_kidney_t1_fix",
        maps={
            "t1_map.nii.gz" : {
                "glob" : "%s/t1_map.nii.gz",
            },
            "t1_conf.nii.gz" : {
                "glob" : "%s/t1_map.nii.gz",
            },
        },
    ),
    seg_postprocess.SegFix(
        "seg_kidney_t1",
        fix_dir_option="seg_kidney_t1_fix",
        segs={
            "*cortex_l*.nii.gz" : {
                "glob" : "%s/seg_kidney_cortex*.nii.gz",
                "side" : "left",
                "fname" : "seg_kidney_cortex_l.nii.gz",
            },
            "*cortex_r*.nii.gz" : {
                "glob" : "%s/seg_kidney_cortex*.nii.gz",
                "side" : "right",
                "fname" : "seg_kidney_cortex_r.nii.gz",
            },
            "*medulla_l*.nii.gz" : {
                "glob" : "%s/seg_kidney_medulla*.nii.gz",
                "side" : "left",
                "fname" : "seg_kidney_medulla_l.nii.gz",
            },
            "*medulla_r*.nii.gz" : {
                "glob" : "%s/seg_kidney_medulla*.nii.gz",
                "side" : "right",
                "fname" : "seg_kidney_medulla_r.nii.gz",
            },
        },
        map_dir="t1_stitched_fix",
        map_fname="t1_map.nii.gz"
    ),
    align.FlirtAlignOnly(
        name="seg_kidney_t1_align_t2star",
        in_dir="t1_stitched_fix",
        in_glob="t1_map.nii.gz",
        ref_dir="t2star",
        ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w",
        weight_mask="kidney_mask.nii.gz",
        weight_mask_dil=6,
        also_align={
            "seg_kidney_t1_fix" : "seg_kidney*.nii.gz",
        }
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_t1_clean",
        srcdir="seg_kidney_t1_align_t2star",
        seg_t1_glob="seg_kidney*.nii.gz",
        t1_map_srcdir="seg_kidney_t1_align_t2star",
        t1_map_glob="t1_map.nii.gz",
        t2w=True
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_t1_clean_native",
        srcdir="seg_kidney_t1_fix",
        seg_t1_glob="seg_kidney*.nii.gz",
        t1_map_srcdir="t1_stitched_fix",
        t1_map_glob="t1_map.nii.gz",
        t2w=True
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_t1_clean_native_generic",
        srcdir="seg_kidney_t1_fix",
        seg_t1_glob="seg_kidney*.nii.gz",
        t1_map_srcdir="t1_stitched_fix",
        t1_map_glob="t1_map.nii.gz",
        t2w=False
    ),
    Stats(),
    statistics.CMD(
        cmd_params=["t2star", "t2star", "t1", "mtr"],
        skip_params=["t1_noclean"],
    ),
    statistics.ShapeMetrics(
        seg_dir="seg_kidney_t2w",
        seg_globs=["*left*.nii.gz", "*right*.nii.gz"],
    ),
    statistics.ISNR(
        src=Module.INPUT,
        imgs={
            "t1w" : "t1w.nii.gz",
            "t2w" : "t2w.nii.gz",
        }
    )
]

def add_options(parser):
    parser.add_argument("--seg-kidney-t1-fix", help="Directory containing manual kidney cortex/medulla masks")
