"""
AIRECON extra shape metrics from fixed masks
"""

from fproc.modules import statistics, seg_postprocess

__version__ = "1.0.0"

NAME = "airecon_fixed_masks"

BATCH = {
    "job_name": "airecon_fproc",
    "time": "02:00:00",
    "partition": "defq",
    "mem_per_cpu": "64g",
    "array": "0-200",
    "input": "/spmstore/project/RenalMRI/airecon//fixed_masks",
    "output": "/spmstore/project/RenalMRI/airecon//output_20260324",
    "input_subfolder": "",
    "output_subfolder": "fproc",
    "modules": ["conda-img"],
    "extra_fproc_args": ["--force", "fixed_masks_shape_metrics"],
}

MODULES = [
    seg_postprocess.SplitLR("", "*.nii.gz", src="INPUT", name="fixed_masks"),
    statistics.ShapeMetrics(
        name="fixed_masks_shape_metrics",
        seg_dir="fixed_masks",
        segs={"tkv_l": "*_l.nii.gz", "tkv_r": "*_r.nii.gz"},
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
    statistics.SegStats(
        name="fixed_masks_vols",
        segs={
            "tkv_all": {
                "dir": "fixed_masks",
                "glob": "*.nii.gz",
            },
            "tkv_l": {
                "dir": "fixed_masks",
                "glob": "*_l.nii.gz",
            },
            "tkv_r": {
                "dir": "fixed_masks",
                "glob": "*_r.nii.gz",
            },
        },
        seg_volumes=True,
    ),
]


def add_options(parser):
    parser.add_argument("--fixed-masks", help="Directory containing fixed masks")
