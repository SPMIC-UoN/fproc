import logging

from fproc.module import Module
from fproc.modules import maps, segmentations, statistics, seg_postprocess

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME="afirm_t2"

class StatsRenalPreproc(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats_renal_preproc",
            segs = {
                "kidney_cortex" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "seg_kidney*_cortex_orig_t1_*.nii.gz",
                },
                "kidney_cortex_l" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "seg_kidney*_cortex_l_orig_t1_*.nii.gz",
                },
                "kidney_cortex_r" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "seg_kidney*_cortex_r_orig_t1_*.nii.gz",
                },
                "kidney_medulla" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "seg_kidney*_medulla_orig_t1_*.nii.gz",
                },
                "kidney_medulla_l" : {
                    "glob" : "seg_kidney*_medulla_l_orig_t1_*.nii.gz",
                },
                "kidney_medulla_r" : {
                    "dir" : "../t1_clean_out",
                    "glob" : "seg_kidney*_medulla_r_orig_t1_*.nii.gz",
                },
                "tkv_l" : {
                    "dir" : "../tkv_fix_out",
                    "glob" : "seg_t2w*_left.nii.gz",
                },
                "tkv_r" : {
                    "dir" : "../tkv_fix_out",
                    "glob" : "seg_t2w*_right.nii.gz",
                },
            },
            params = {
                "t2star_exp" : {
                    "dir " : "../t2star_out",
                    "glob" : "map_t2star_*_exp.nii.gz",
                },
                "t2star_loglin" : {
                    "dir " : "../t2star_out",
                    "glob" : "map_t2star_*_loglin.nii.gz",
                },
                "r2star_exp" : {
                    "dir " : "../t2star_out",
                    "glob" : "map_r2star_*_exp.nii.gz",
                },
                "r2star_loglin" : {
                    "dir " : "../t2star_out",
                    "glob" : "map_r2star_*_loglin.nii.gz",
                },
                "t1" : {
                    "dir" : "../t1_out",
                    "glob" : "map_t1_conf_*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex" : {"dir" : "../t1_clean_native2_out"},
                        "kidney_cortex_r" : {"dir" : "../t1_clean_native2_out"},
                        "kidney_cortex_l" : {"dir" : "../t1_clean_native2_out"},
                        "kidney_medulla" : {"dir" : "../t1_clean_native2_out"},
                        "kidney_medulla_l" : {"dir" : "../t1_clean_native2_out"},
                        "kidney_medulla_r" : {"dir" : "../t1_clean_native2_out"},
                    },
                },
                "t1_noclean" : {
                    "dir" : "../t1_out",
                    "glob" : "map_t1_conf_*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex" : {"dir" : "../t1_clean_native_out"},
                        "kidney_cortex_r" : {"dir" : "../t1_clean_native_out"},
                        "kidney_cortex_l" : {"dir" : "../t1_clean_native_out"},
                        "kidney_medulla" : {"dir" : "../t1_clean_native_out"},
                        "kidney_medulla_l" : {"dir" : "../t1_clean_native_out"},
                        "kidney_medulla_r" : {"dir" : "../t1_clean_native_out"},
                    }
                },
                "mtr" : {
                    "dir" : "../mtr_out",
                    "glob" : "map_mtr_*.nii.gz",
                },
                "b0" : {
                    "dir" : "../b0_out",
                    "glob" : "map_b0_*.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"],
                },
                "b1" : {
                    "dir" : "../b1_out",
                    "glob" : "map_b1_*_noscale.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"],
                },
                "b1_rescaled" : {
                    "dir" : "../b1_out",
                    "glob" : "map_b1_*_rescaled.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"],
                },
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
        )

class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats",
            default_limits="3t",
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

class StatsDixon(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats_dixon", 
            default_limits="3t",
            segs={
                "liver_ax" : {
                    "dir" : "seg_liver_dixon_ax",
                    "glob" : "liver.nii.gz"
                },
                "spleen_ax" : {
                    "dir" : "seg_spleen_dixon_ax",
                    "glob" : "spleen.nii.gz"
                },
                "sat_ax" : {
                    "dir" : "seg_sat_dixon_ax",
                    "glob" : "sat.nii.gz",
                },
                "liver_cor" : {
                    "dir" : "seg_liver_dixon_cor",
                    "glob" : "liver.nii.gz"
                },
                "spleen_cor" : {
                    "dir" : "seg_spleen_dixon_cor",
                    "glob" : "spleen.nii.gz"
                },
                "sat_cor" : {
                    "dir" : "seg_sat_dixon_cor",
                    "glob" : "sat.nii.gz",
                },
                "pancreas" : {
                    "dir" : "seg_pancreas_ethrive",
                    "glob" : "pancreas.nii.gz",
                },
            },
            params={
                "t2star_exp" : {
                    "dir" : "../t2star_out",
                    "glob" : "map_t2star*_exp.nii.gz",
                },
                "r2star_exp" : {
                    "dir" : "../t2star_out",
                    "glob" : "map_r2star*_exp.nii.gz",
                },
                "t2star_loglin" : {
                    "dir" : "../t2star_out",
                    "glob" : "map_t2star*_loglin.nii.gz",
                },
                "r2star_loglin" : {
                    "dir" : "../t2star_out",
                    "glob" : "map_r2star*_loglin.nii.gz",
                },
                "t1" : {
                    "dir" : "../t1_out",
                    "glob" : "map_t1_conf_*.nii.gz",
                },
                "t1_ax" : {
                    "dir" : "molli_ax",
                    "src" : self.INPUT,
                    "glob" : "t1_conf.nii.gz",
                },
                "mtr" : {
                    "dir" : "../mtr_out",
                    "glob" : "map_mtr_*.nii.gz",
                },
                "b0" : {
                    "dir" : "../b0_out",
                    "glob" : "map_b0_*.nii.gz",
                },
                "b1" : {
                    "dir" : "../b1_out",
                    "glob" : "map_b1_*_noscale.nii.gz",
                },
                "b1_rescaled" : {
                    "dir" : "../b1_out",
                    "glob" : "map_b1_*_rescaled.nii.gz",
                },
                "t2_exp" : {
                    "dir" : "t2",
                    "glob" : "t2_exp.nii.gz",
                },
                "t2_stim" : {
                    "dir" : "t2",
                    "glob" : "t2_stim.nii.gz",
                },
                "b1_exp" : {
                    "dir" : "t2",
                    "glob" : "b1_exp.nii.gz",
                },
                "b1_stim" : {
                    "dir" : "t2",
                    "glob" : "b1_stim.nii.gz",
                },
                "ff_cor" : {
                    "dir" : "ff_dixon_cor",
                    "glob" : "fat_fraction_orig.nii.gz",
                },
                "ff_ax" : {
                    "dir" : "ff_dixon_ax",
                    "glob" : "fat_fraction_orig.nii.gz",
                },
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
            seg_volumes=False,
        )

MODULES = [
    maps.T2(),
    maps.FatFractionDixon(name="ff_dixon_cor", dixon_dir="dixon_cor"),
    maps.FatFractionDixon(name="ff_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.KidneyCystT2w(t2w_dir="../tkv_out", t2w_glob="map_t2w*.nii.gz", t2w_src=Module.INPUT),
    seg_postprocess.KidneyCystClean(seg_t2w_dir="../tkv_fix_out"),
    #segmentations.BodyDixon(),
    segmentations.SatDixon(name="seg_sat_dixon_cor", dixon_dir="dixon_cor"),
    segmentations.LiverDixon(name="seg_liver_dixon_cor", dixon_dir="dixon_cor"),
    segmentations.SpleenDixon(name="seg_spleen_dixon_cor",dixon_dir="dixon_cor"),
    segmentations.SatDixon(name="seg_sat_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.LiverDixon(name="seg_liver_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.SpleenDixon(name="seg_spleen_dixon_ax", dixon_dir="dixon_ax"),
    #segmentations.KidneyDixon(model_id="422"),
    segmentations.PancreasEthrive(),
    Stats(),
    StatsDixon(),
]
