import logging

from fproc.module import Module, StatsModule
from fproc.modules import maps, segmentations

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME="afirm_t2"

class Stats(StatsModule):
    def __init__(self):
        StatsModule.__init__(
            self, name="stats", 
            segs={
                "cortex_r" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "*cortex_r*.nii.gz"
                },
                "cortex_l" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "*cortex_l*.nii.gz"
                },
                
                "medulla_r" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "*medulla_r*.nii.gz"
                },
                "medulla_l" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "*medulla_l*.nii.gz"
                },
                "kidney_l" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "*all_l_orig*.nii.gz"
                },
                "kidney_r" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "*all_r_orig*.nii.gz"
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
                }
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
            seg_volumes=False,
        )

MODULES = [
    maps.T2(),
    segmentations.KidneyCystT2w(t2w_dir="../tkv_out", t2w_glob="map_t2w*.nii.gz", t2w_src=Module.INPUT),
    Stats(),
]
