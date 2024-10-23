import logging
import sys

import numpy as np

from fproc.module import Module
from fproc.modules import maps, statistics
from fsort.image_file import ImageFile

LOG = logging.getLogger(__name__)

class SegLegDixon(Module):
    def __init__(self, name="seg_leg_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        fat_glob = self.kwargs.get("fat_glob", "fat.nii.gz")
        water_glob = self.kwargs.get("water_glob", "water.nii.gz")
        
        fat = self.single_inimg(dixon_dir, fat_glob, src=self.INPUT)
        water = self.single_inimg(dixon_dir, water_glob, src=self.INPUT)
        if fat is None or water is None:
            self.no_data(f"Could not find fat/water matching {fat_glob} {water_glob}")

        fat.save(self.outfile("fat.nii.gz"))
        water.save(self.outfile("fat.nii.gz"))
        outfile = self.outfile("leg.nii.gz")
        retval = self.runcmd([
            "leg_seg_dixon",
            fat.fpath, water.fpath, outfile,
        ], logfile=self.outfile("seg.log"))
        
        if retval == 0:
            seg = ImageFile(outfile, warn_json=False)
            regions = {
                "calf_muscle_r" : [1],
                "calf_sat_r" : [2],
                "calf_muscle_l" : [3],
                "calf_sat_l" : [4],
                "thigh_muscle_r" : [5],
                "thigh_sat_r" : [6],
                "thigh_muscle_l" : [7],
                "thigh_sat_l" : [8],
                "calf_muscle" : [1, 3],
                "calf_sat" : [2, 4],
                "thigh_muscle" : [5, 7],
                "thigh_sat" : [6, 8],
                "muscle_r" : [1, 5],
                "sat_r" : [2, 6],
                "muscle_l" : [3, 7],
                "sat_l" : [4, 8],
                "total" : [2, 4, 6, 8],
            }
            for name, regions in regions.items():
                mask = np.zeros_like(seg.data, dtype=np.int8)
                for idx in regions:
                    mask[seg.data == idx] = 1
                fname = self.outfile(f"{name}.nii.gz")
                seg.save_derived(mask, fname)
                mask = ImageFile(fname, warn_json=False)
                if "sat" in name:
                    self.lightbox(fat, mask, f"{name}_fat")
                else:
                    self.lightbox(water, mask, f"{name}_water")

class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats", 
            segs={
                "calf_muscle_r" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "calf_muscle_r.nii.gz"
                },
                "calf_muscle_l" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "calf_muscle_l.nii.gz"
                },
                "thigh_muscle_r" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "thigh_muscle_r.nii.gz"
                },
                "thigh_muscle_l" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "thigh_muscle_l.nii.gz"
                },
                "calf_sat_r" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "calf_sat_r.nii.gz"
                },
                "calf_sat_l" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "calf_sat_l.nii.gz"
                },
                "thigh_sat_r" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "thigh_sat_r.nii.gz"
                },
                "thigh_sat_l" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "thigh_sat_l.nii.gz"
                },
                "calf_muscle" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "calf_muscle.nii.gz"
                },
                "calf_sat" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "calf_sat.nii.gz"
                },
                "thigh_muscle" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "thigh_muscle.nii.gz"
                },
                "thigh_sat" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "thigh_sat.nii.gz"
                },
                "muscle_r" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "muscle_r.nii.gz"
                },
                "sat_r" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "sat_r.nii.gz"
                },
                "muscle_l" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "muscle_l.nii.gz"
                },
                "sat_l" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "sat_l.nii.gz"
                },
                "total" : {
                    "dir" : "seg_leg_dixon",
                    "glob" : "total.nii.gz"
                },
            },
            params={
                "ff" : {
                    "dir" : "fat_fraction",
                    "glob" : "ff_orig.nii.gz",
                    "limits" : (0, 1),
                },
                "t2star" : {
                    "dir" : "t2star_dixon",
                    "glob" : "t2star_exclude_fill.nii.gz",
                    "limits" : (2, 100),
                },
            },
            stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

__version__ = "0.0.1"

NAME = "wbdixon"

MODULES = [
    # Maps
    maps.FatFractionDixon(dixon_dir="", ff_name="ff"),
    maps.T2starDixon(dixon_dir=""),
    # Segmentations
    SegLegDixon(dixon_dir=""),
    # Statistics
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--leg-dixon-model", help="Filename or URL for segmentation model weights")
