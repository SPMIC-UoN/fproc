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
                "cortex_r" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "*cortex_r*.nii.gz"
                },
                "cortex_l" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "*cortex_l*.nii.gz"
                },
                # "cortex" : {
                #     "dir" : "t1_segs",
                #     "glob" : "*cortex*.nii.gz"
                # },
                "medulla_r" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "*medulla_r*.nii.gz"
                },
                "medulla_l" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "*medulla_l*.nii.gz"
                },
                # "medulla" : {
                #     "dir" : "t1_segs",
                #     "glob" : "*medulla*.nii.gz"
                # },
                "kidney_l" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "*all_l_*.nii.gz"
                },
                "kidney_r" : {
                    "dir" : "seg_kidney_t1",
                    "glob" : "*all_r_*.nii.gz"
                },
            },
            params={
                "t2_exp" : {
                    "dir" : "t2",
                    "glob" : "t2_exp.nii.gz",
                    "segs" : ["cortex_r", "cortex_l", "medulla_r", "medulla_l"],
                },
                "t2_stim" : {
                    "dir" : "t2",
                    "glob" : "t2_stim.nii.gz",
                    "segs" : ["cortex_r", "cortex_l", "medulla_r", "medulla_l"],
                },
                "b1_stim" : {
                    "dir" : "t2",
                    "glob" : "b1_stim.nii.gz",
                },
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
                    "dir" : "t1_kidney",
                    "glob" : "t1_map*.nii.gz",
                },
                "mtr" : {
                    "dir" : "mtr",
                    "glob" : "mtr.nii.gz",
                },
                "b0" : {
                    "dir" : "b0",
                    "glob" : "b0.nii.gz",
                },
                "b1" : {
                    "dir" : "b1",
                    "glob" : "b1.nii.gz",
                },
                "b1" : {
                    "dir" : "b1",
                    "glob" : "b1_rescaled.nii.gz",
                }
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
            seg_volumes=False,
        )

NAME="tk21_t2"

MODULES = [
    CopyModule("t1w"),
    T1Kidney(),
    maps.B0(),
    maps.B1(),
    maps.MTR(),
    maps.T2star(expected_echos=12),
    maps.T2(methods=["exp", "stim"], echos=10, max_echos=11),
    segmentations.KidneyT1(),
    segmentations.KidneyT2w(),
    seg_postprocess.KidneyT1Clean(),
    seg_postprocess.SegFix(
        src_dir="seg_kidney_t1_clean", src_glob="kidney*.nii.gz", 
        fix_dir_option="kidney_masks", fix_glob="%s/*mask*.nii*"
    ),
    #align.FlirtAlignOnly("seg_kidney_t1_clean_fix", )
    Stats(),
    statistics.CMD(),
    statistics.ShapeMetrics("seg_kidney_t2w", "kidney*.nii.gz"),
]

def add_options(parser):
    parser.add_argument("--t2star-method", help="Method to use when doing T2* processing", choices=["loglin", "2p_exp", "all"], default="all")
    parser.add_argument("--t2star-resample", help="Planar resolution to resample T2* to (mm)", type=float, default=0)
    parser.add_argument("--kidney-t2w-model", "--segmentation-weights", help="Filename or URL for T2w segmentation CNN weights")
    parser.add_argument("--kidney-t1-model", help="Filename or URL for T1 segmentation model weights")
    parser.add_argument("--kidney-masks", help="Directory containing manual kidney cortex/medulla masks")
