import logging

from fproc.modules import statistics, maps, seg_postprocess

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME = "normative_tk21_diff_mtr"

class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats", 
            segs={
                "kidney_cortex_l" : {
                    "dir" : "masks_splitlr",
                    "glob" : "cortex_l.nii.gz"
                },
                "kidney_cortex_r" : {
                    "dir" : "masks_splitlr",
                    "glob" : "cortex_r.nii.gz"
                },
                "kidney_medulla_l" : {
                    "dir" : "masks_splitlr",
                    "glob" : "medulla_l.nii.gz"
                },
                "kidney_medulla_r" : {
                    "dir" : "masks_splitlr",
                    "glob" : "medulla_r.nii.gz"
                },
            },
            params={
                "d" : {
                    "dir" : "diffusion_data",
                    "glob" : "d_map.nii.gz",
                    "limits" : (1, 3),
                },
                "dstar" : {
                    "dir" : "diffusion_data",
                    "glob" : "dstar_map.nii.gz",
                    "limits" : (None, 15),
                },
                "pf" : {
                    "dir" : "diffusion_data",
                    "glob" : "pf_map.nii.gz",
                    "limits" : (None, 0.5),
                },
                "mtr" : {
                    "dir" : "asl_mtr",
                    "glob" : "asl_mtr.nii.gz",
                },
            },
            stats=["iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

MODULES = [
    maps.AdditionalMap(
        "asl_mtr",
        srcdir_option="datadir",
        maps={
            "asl_mtr.nii.gz" : "asl_mtr/%s/mtr/reg_mtr_map.nii.gz"
        }
    ),
    maps.AdditionalMap(
        "diffusion_data",
        srcdir_option="datadir",
        maps={
            "d_map.nii.gz" : "diffusion_data/%s/*Dmap.nii.gz",
            "dstar_map.nii.gz" : "diffusion_data/%s/*DSTARmap.nii.gz",
            "pf_map.nii.gz" : "diffusion_data/%s/*PFmap.nii.gz",
            "adc_map.nii.gz" : "diffusion_data/%s/*ADCmap.nii.gz",
        }
    ),
    maps.AdditionalMap(
        "masks",
        srcdir_option="datadir",
        maps={
            "cortex.nii.gz" : "masks/%s/*cortex*.nii.gz",
            "medulla.nii.gz" : "masks/%s/*medulla*.nii.gz",
        }
    ),
    seg_postprocess.SplitLR("masks", "*.nii.gz"),
    Stats(),
]

def add_options(parser):
    parser.add_argument("--datadir", help="Input data dir")
