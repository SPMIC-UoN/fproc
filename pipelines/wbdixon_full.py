import logging
import sys

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation

from fproc.module import Module
from fproc.modules import maps, statistics, segmentations, seg_postprocess
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

        fat_reorient_nii = nib.as_closest_canonical(fat.nii).as_reoriented(np.array([[0, 1], [1, 1], [2, 1]]))
        fat_reorient_nii.to_filename(self.outfile("fat.nii.gz"))
        water_reorient_nii = nib.as_closest_canonical(water.nii).as_reoriented(np.array([[0, 1], [1, 1], [2, 1]]))
        water_reorient_nii.to_filename(self.outfile("water.nii.gz"))
        fat = ImageFile(self.outfile("fat.nii.gz"), warn_json=False)
        water = ImageFile(self.outfile("water.nii.gz"), warn_json=False)

        #fat.save(self.outfile("fat.nii.gz"))
        #water.save(self.outfile("water.nii.gz"))
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
                if "muscle" in name:  # Apply dilation only to muscle masks
                    LOG.info(f"Applying dilation to {name}")
                    seg.save_derived(mask, self.outfile(f"{name}_nodil.nii.gz"))
                    mask = binary_dilation(mask)
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
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "calf_muscle_r.nii.gz"
                },
                "calf_muscle_l" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "calf_muscle_l.nii.gz"
                },
                "thigh_muscle_r" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "thigh_muscle_r.nii.gz"
                },
                "thigh_muscle_l" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "thigh_muscle_l.nii.gz"
                },
                "calf_sat_r" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "calf_sat_r.nii.gz"
                },
                "calf_sat_l" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "calf_sat_l.nii.gz"
                },
                "thigh_sat_r" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "thigh_sat_r.nii.gz"
                },
                "thigh_sat_l" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "thigh_sat_l.nii.gz"
                },
                "calf_muscle" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "calf_muscle*.nii.gz"
                },
                "calf_sat" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "calf_sat*.nii.gz"
                },
                "thigh_muscle" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "thigh_muscle*.nii.gz"
                },
                "thigh_sat" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "thigh_sat*.nii.gz"
                },
                "muscle_r" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "*muscle_r.nii.gz"
                },
                "sat_r" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "*sat_r.nii.gz"
                },
                "muscle_l" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "*muscle_l.nii.gz"
                },
                "sat_l" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "*sat_l.nii.gz"
                },
                "total" : {
                    "dir" : "seg_leg_dixon_fix",
                    "glob" : "*.nii.gz"
                },
            },
            params={
                "ff_scanner" : {
                    "dir" : "fat_fraction",
                    "glob" : "ff_scanner.nii.gz",
                    "limits" : (0, 100),
                },
                "ff_calc" : {
                    "dir" : "fat_fraction",
                    "glob" : "ff_calc.nii.gz",
                    "limits" : (0, 100),
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
    maps.DixonClassify(
        name="dixon_classify_1",
        model="/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier_20250626.h5",
        dixon_src="raw_dixon",
        dixon_glob="raw_dixon_series_1*.nii.gz",
    ),
    maps.DixonClassify(
        name="dixon_classify_2",
        model="/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier_20250626.h5",
        dixon_src="raw_dixon",
        dixon_glob="raw_dixon_series_2*.nii.gz",
    ),
    maps.DixonClassify(
        name="dixon_classify_3",
        model="/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier_20250626.h5",
        dixon_src="raw_dixon",
        dixon_glob="raw_dixon_series_3*.nii.gz",
    ),
    maps.DixonClassify(
        name="dixon_classify_4",
        model="/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier_20250626.h5",
        dixon_src="raw_dixon",
        dixon_glob="raw_dixon_series_4*.nii.gz",
    ),
    maps.DixonClassify(
        name="dixon_classify_5",
        model="/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier_20250626.h5",
        dixon_src="raw_dixon",
        dixon_glob="raw_dixon_series_5*.nii.gz",
    ),
    maps.DixonClassify(
        name="dixon_classify_6",
        model="/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier_20250626.h5",
        dixon_src="raw_dixon",
        dixon_glob="raw_dixon_series_6*.nii.gz",
    ),
]


def add_options(parser):
    parser.add_argument("--fixed-masks", help="Directory containing manual masks")
