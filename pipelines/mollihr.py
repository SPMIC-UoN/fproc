from collections import OrderedDict
import logging

import numpy as np

from fproc.module import Module, CopyModule
from fproc.modules import maps, segmentations, statistics, seg_postprocess

LOG = logging.getLogger(__name__)

class T1(Module):
    def __init__(self):
        Module.__init__(self, "t1")

    def process(self):
        t1_maps = self.inimgs("molli", "t1_map_*.nii.gz", src=self.INPUT)
        if not t1_maps:
            self.no_data("No T1 maps found")

        for t1_map in t1_maps:
            t1_map.save(self.outfile(t1_map.fname))
            LOG.info(f" - Found {t1_map.fname}")
            conf_fname = t1_map.fname.replace("map", "conf")
            t1_conf = self.inimg("molli",conf_fname , src=self.INPUT)
            t1_conf.save(self.outfile(conf_fname))

class MolliRaw5(maps.T1Molli):
    def __init__(self):
        Module.__init__(self, "molli_raw_5")

    def process(self):
        molli_raw = self.inimgs("molli_raw", "molli_raw*.nii.gz", src=self.INPUT)
        if not molli_raw:
            self.no_data("No raw MOLLI data found")

        for molli in molli_raw:
            if molli.nvols != 7:
                LOG.warn(f"{molli.fname} has {molli.nvols} volumes - expected 7. Ignoring")
                continue
            if not molli.inversiontimedelay or len(molli.inversiontimedelay) < 7:
                LOG.warn(f"{molli.fname} did not contain at least 7 inversion times: {molli.inversiontimedelay} - ignoring")
                continue

            data = np.stack([molli.data[..., v] for v in [0, 3, 4, 5, 6]], axis=-1)
            tis = [molli.inversiontimedelay[t] for t in [0, 3, 4, 5, 6]]
            molli.metadata["InversionTimeDelay"] = tis
            LOG.info(f" - {molli.fname} selecting TIs: {tis}")
            molli.save_derived(data, self.outfile(molli.fname.replace(".nii.gz", "_5.nii.gz")))

class SegStats(statistics.SegStats):
    def __init__(self):
        segs = {
            "liver" : {
                "dir" : "seg_liver_dixon",
                "glob" : "liver.nii.gz"
            },
            "spleen" : {
                "dir" : "seg_spleen_dixon",
                "glob" : "spleen.nii.gz"
            },
            "kidney_dixon" : {
                "dir" : "seg_kidney_dixon",
                "glob" : "kidney.nii.gz",
            },
            "kidney_cortex_l" : {
                "dir" : "seg_kidney_t1_clean",
                "glob" : f"kidney_t1_map_*_cortex_l_t1.nii.gz"
            },
            "kidney_cortex_r" : {
                "dir" : "seg_kidney_t1_clean",
                "glob" : f"kidney_t1_map_*_cortex_r_t1.nii.gz"
            },
            "kidney_medulla_l" : {
                "dir" : "seg_kidney_t1_clean",
                "glob" : f"kidney_t1_map_*_medulla_l_t1.nii.gz"
            },
            "kidney_medulla_r" : {
                "dir" : "seg_kidney_t1_clean",
                "glob" : f"kidney_t1_map_*_medulla_r_t1.nii.gz"
            },
        }

        params = OrderedDict()
        params["ff"] = {
            "dir" : "fat_fraction",
            "glob" : "fat_fraction_orig.nii.gz",
            "limits" : (0, 1),
            "segs" : list(segs.keys())
        }
        params["t2star"] = {
            "dir" : "t2star",
            "glob" : "t2star_exclude_fill.nii.gz",
            "limits" : (2, 100),
            "segs" : list(segs.keys())
        }
        params["b0"] = {
            "dir" : "b0",
            "glob" : "b0.nii.gz",
            "segs" : list(segs.keys())
        }
        params["b1"] = {
            "dir" : "b1",
            "glob" : "b1.nii.gz",
            "segs" : list(segs.keys())
        }

        for idx in range(1, 21):
            segs.update({
                f"kidney_cortex_l_{idx}" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : f"kidney_t1_map_{idx}_cortex_l_t1.nii.gz"
                },
                f"kidney_cortex_r_{idx}" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : f"kidney_t1_map_{idx}_cortex_r_t1.nii.gz"
                },
                f"kidney_medulla_l_{idx}" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : f"kidney_t1_map_{idx}_medulla_l_t1.nii.gz"
                },
                f"kidney_medulla_r_{idx}" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : f"kidney_t1_map_{idx}_medulla_r_t1.nii.gz"
                },
            })

            params[f"t1_{idx}"] = {
                "dir" : "t1",
                "glob" : f"t1_map_{idx}.nii.gz",
                "segs" : [
                    f"kidney_cortex_l_{idx}",
                    f"kidney_cortex_r_{idx}",
                    f"kidney_medulla_l_{idx}",
                    f"kidney_medulla_r_{idx}",
                    "spleen",
                    "liver",
                ]
            }

        for idx in range(1, 21):
            params[f"t1_molli_5tis_{idx}"] = {
                "dir" : "t1_molli_5",
                "glob" : f"molli_raw_{idx}_5_t1_map.nii.gz",
                "segs" : [
                    f"kidney_cortex_l_{idx}",
                    f"kidney_cortex_r_{idx}",
                    f"kidney_medulla_l_{idx}",
                    f"kidney_medulla_r_{idx}",
                    "spleen",
                    "liver",
                ]
            }

        for idx in range(1, 21):
            params[f"t1_molli_7tis_{idx}"] = {
                "dir" : "t1_molli_7",
                "glob" : f"molli_raw_{idx}_t1_map.nii.gz",
                "segs" : [
                    f"kidney_cortex_l_{idx}",
                    f"kidney_cortex_r_{idx}",
                    f"kidney_medulla_l_{idx}",
                    f"kidney_medulla_r_{idx}",
                    "spleen",
                    "liver",
                ]
            }

        statistics.SegStats.__init__(
            self, name="stats", 
            segs=segs,
            params=params,
            stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

__version__ = "0.0.1"

NAME = "mollihr"

MODULES = [
    # Segmentations
    segmentations.LiverDixon(),
    segmentations.SpleenDixon(),
    segmentations.KidneyDixon(),
    segmentations.KidneyT1(map_dir="../fsort/molli"),
    # Parameter maps
    maps.B0(),
    maps.FatFractionDixon(),
    CopyModule("b1"),
    maps.T2starDixon(),
    T1(),
    MolliRaw5(),
    maps.T1Molli(name="t1_molli_5", molli_dir="molli_raw_5", molli_src=Module.OUTPUT),
    maps.T1Molli(name="t1_molli_7"),
    # Segmentation cleaning
    seg_postprocess.KidneyT1Clean(t2w=False),
    # Statistics
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--kidney-t1-model", help="Filename or URL for T1 segmentation model weights")
