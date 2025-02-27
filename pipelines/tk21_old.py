import logging

from fsort import ImageFile
from fproc.module import Module, CopyModule
from fproc.modules import maps, statistics, segmentations, seg_postprocess, align

import numpy as np

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class T1Kidney(Module):
    def __init__(self):
        Module.__init__(self, "t1_kidney")

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
            LOG.info(f" - Copying T1 map/confidence images")
            self.copyinput("t1", "t1_map*.nii.gz")
            self.copyinput("t1", "t1_conf*.nii.gz")

class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats", 
            segs={
                "kidney_cortex" : {
                    "dir" : "seg_kidney_t1_clean_fix",
                    "glob" : "*cortex_t1.nii.gz"
                },
                "kidney_cortex_l" : {
                    "dir" : "seg_kidney_t1_clean_fix",
                    "glob" : "*cortex_l_t1.nii.gz"
                },
                "kidney_cortex_r" : {
                    "dir" : "seg_kidney_t1_clean_fix",
                    "glob" : "*cortex_r_t1.nii.gz"
                },
                "kidney_medulla" : {
                    "dir" : "seg_kidney_t1_clean_fix",
                    "glob" : "*medulla_t1.nii.gz"
                },
                "kidney_medulla_l" : {
                    "dir" : "seg_kidney_t1_clean_fix",
                    "glob" : "*medulla_l_t1.nii.gz"
                },
                "kidney_medulla_r" : {
                    "dir" : "seg_kidney_t1_clean_fix",
                    "glob" : "*medulla_r_t1.nii.gz"
                },
                "tkv_l" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "*left*.nii.gz",
                },
                "tkv_r" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "*right*.nii.gz"
                },
            },
            params={
                "t2star_exp" : {
                    "dir" : "t2star",
                    "glob" : "t2star_2p_exp.nii.gz",
                },
                "t2star_loglin" : {
                    "dir" : "t2star",
                    "glob" : "t2star_loglin.nii.gz",
                },
                "r2star_exp" : {
                    "dir" : "t2star",
                    "glob" : "r2star_2p_exp.nii.gz",
                },
                "r2star_loglin" : {
                    "dir" : "t2star",
                    "glob" : "r2star_loglin.nii.gz",
                },
                "t1" : {
                    "dir" : "t1_molli",
                    "glob" : "*t1_map*.nii.gz",
                },
                "mtr" : {
                    "dir" : "mtr",
                    "glob" : "mtr.nii.gz",
                },
                "b0" : {
                    "dir" : "b0",
                    "glob" : "b0.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"]
                },
                "b1" : {
                    "dir" : "b1",
                    "glob" : "b1.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"]
                },
                "b1_rescaled" : {
                    "dir" : "b1",
                    "glob" : "b1_rescaled.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"]
                },
                "t2_exp" : {
                    "dir" : "t2",
                    "glob" : "t2_exp.nii.gz",
                    "segs" : ["kidney_cortex", "kidney_cortex_l", "kidney_cortex_r", "kidney_medulla", "kidney_medulla_l", "kidney_medulla_r"],
                },
                "t2_stim" : {
                    "dir" : "t2",
                    "glob" : "t2_stim.nii.gz",
                    "segs" : ["kidney_cortex", "kidney_cortex_l", "kidney_cortex_r", "kidney_medulla", "kidney_medulla_l", "kidney_medulla_r"],
                },
                "b1_stim" : {
                    "dir" : "t2",
                    "glob" : "b1_stim.nii.gz",
                },
            },
            stats=["n", "iqn", "iqmean", "median", "iqstd", "vol", "iqvol"],
            seg_volumes=False,
        )

NAME="tk21"

MODULES = [
    CopyModule("t1w"),
    maps.T1Molli(molli_dir="t1", molli_glob="t1_raw_molli*.nii.gz", tis=[235, 315, 1235, 1315, 2235, 2315, 3235, 4235]),
    maps.B0(),
    maps.B1(),
    maps.MTR(),
    maps.T2star(expected_echos=12),
    maps.T2(methods=["exp", "stim"], echos=10, max_echos=11),
    segmentations.KidneyT1(),
    segmentations.KidneyT2w(),
    align.FlirtAlignOnly(
        in_dir="t1_kidney", in_glob="*t1_map*.nii.gz",
        ref_dir="t2star", ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w", weight_mask="kidney_mask.nii.gz", weight_mask_dil=6,
        also_align={
            "seg_kidney_t1" : "*.nii.gz",
        }
    ),
    seg_postprocess.SegFix(
        seg_dir="seg_kidney_t1", src_glob="kidney*.nii.gz", 
        fix_dir_option="kidney_masks", fix_glob="%s/*mask*.nii*",
        segs = {
            "*cortex_l_t1*" : "*eft_cortex*",
            "*cortex_r_t1*" : "*ight_cortex*",
            "*cortex_t1*" : "*cortex_t1**",
            "*medulla_l_t1*" : "*eft_medulla*",
            "*medulla_r_t1*" : "*ight_medulla*",
            "*medulla_t1*" : "*medulla_t1**",
        },
        map_dir="seg_kidney_t1",
        map_fname="map_t1*nii.gz",
    ),
    seg_postprocess.KidneyT1Clean(),
    Stats(),
    statistics.CMD(cmd_params=["t2star_exp", "t2star_loglin", "r2star_exp", "r2star_loglin", "t1", "mtr"]),
    statistics.ShapeMetrics(seg_dir="seg_kidney_t2w", seg_globs=["kidney*left*.nii.gz", "kidney*right*.nii.gz"]),
    seg_postprocess.SegVolumes(name="kidney_volumes", seg_dir="seg_kidney_t2w", vol_fname="tkv.csv", segs={
        "kv_left" : "kidney_left_kidney.nii.gz", 
        "kv_right" : "kidney_mask.nii.gz", 
        "kv_total" :  "kidney_right_kidney.nii.gz"
    })
]

def add_options(parser):
    parser.add_argument("--kidney-masks", help="Directory containing manual kidney cortex/medulla masks")
