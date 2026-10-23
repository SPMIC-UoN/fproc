import logging
import os
import re
import shutil

from scipy import ndimage
import numpy as np

from fproc.module import Module
from fproc.modules import (
    maps,
    statistics,
    regrid,
    misc,
)

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

# Configuration
NAME = "phantoms"
STUDYDIR = os.path.join("/gpfs01/spmstore/project/RenalMRI", NAME)
OUTNAME = NAME

BATCH = {
    "time": "2:00:00",
    "mem_per_cpu": "32g",
    "subjects": 100,
}

COHORTS = [
    ("full cohort", "", None),
]

OUTFILES = {
    "stats": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/stats/stats.csv",
    ],
}


def add_options(parser):
    parser.add_argument(
        "--rois-dir",
        required=True,
        help="Directory containing subject-matched pre-drawn ROI files",
    )


class CopyPreDrawnROIs(Module):
    """Copy matching pre-drawn ROI images into the module output folder."""

    ROI_FILENAME = re.compile(r"^voi\d+\.nii\.gz$")

    def __init__(self, name="pre_drawn_rois", **kwargs):
        super().__init__(name, **kwargs)

    def process(self):
        rois_dir = self.pipeline.options.rois_dir
        if not os.path.isdir(rois_dir):
            self.bad_data(f"ROIs directory does not exist: {rois_dir}")

        subjid = self.pipeline.options.subjid
        matching_dirs = sorted(
            os.path.join(rois_dir, name)
            for name in os.listdir(rois_dir)
            if os.path.isdir(os.path.join(rois_dir, name))
            and name.casefold() in subjid.casefold()
        )
        if not matching_dirs:
            LOG.warning(" - No ROI subdirectories matched subject ID %s", subjid)
            return

        roi_files = {}
        for matching_dir in matching_dirs:
            for root, dirs, filenames in os.walk(matching_dir):
                dirs.sort()
                for filename in sorted(filenames):
                    if self.ROI_FILENAME.fullmatch(filename):
                        roi_files.setdefault(filename, []).append(
                            os.path.join(root, filename)
                        )

        if not roi_files:
            LOG.warning(
                " - No voi<n>.nii.gz files found for subject ID %s in %s",
                subjid,
                rois_dir,
            )
            return

        for filename in sorted(roi_files):
            sources = sorted(roi_files[filename])
            if len(sources) > 1:
                LOG.warning(
                    " - Found %d files named %s; using %s and ignoring additional matches: %s",
                    len(sources),
                    filename,
                    sources[0],
                    ", ".join(sources[1:]),
                )
            shutil.copy2(sources[0], self.outfile(filename))


class T1DataSmoothed(Module):
    """
    To MOLLI Philips (DER/NOT/NEW) and GE (CBG) ONLY: Add spatial smoothing of 2 mm (so -s 0.85 in fsl) 
    to both the scanner generated maps and raw data with fits re-run model, and segmentations done on 
    smoothed data to estimate metrics from smoothed data
    """

    def __init__(self, name="t1_data_smoothed", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dir = self.kwargs.get("dir", "t1_molli_raw")
        glob = self.kwargs.get("glob", "t1_molli_raw*.nii.gz")
        t1_imgs = self.inimgs(dir, glob)
        if not t1_imgs:
            self.no_data(f"No raw MOLLI data found in {dir}/{glob}")
        
        for img in t1_imgs:
            if img.vendor in ("philips", "ge"):
                LOG.info(f" - Smoothing raw MOLLI data for {img.fname} (vendor: {img.vendor})") 
                sigma = [0.85 / v for v in img.nii.header.get_zooms()[:2]]
                while len(sigma) < img.data.ndim:
                    sigma.append(0)

                smoothed_data = ndimage.gaussian_filter(img.data, sigma=sigma)
            else:
                LOG.info(f" - Skipping smoothing for {img.fname} (vendor: {img.vendor})")
                smoothed_data = img.data
            img.save_derived(smoothed_data, self.outfile(img.fname))

MODULES = [
    misc.ScanDates(
        "scan_dates",
        input={
            "../fsort/t1w": "*.nii.gz",
            "../fsort/t2w": "*.nii.gz",
            "../fsort/t1_molli": "*.nii.gz",
        },
    ),
    # Parameter maps
    T1DataSmoothed(name="t1_molli_raw_smoothed", dir="../fsort/t1_molli_raw", glob="t1_molli_raw*.nii.gz"),
    T1DataSmoothed(name="t1_molli_smoothed", dir="../fsort/t1_molli", glob="t1_*.nii.gz"),
    maps.T1Molli(
        name="t1_molli",
        molli_dir="../fproc/t1_molli_smoothed",
        molli_glob="t1_molli_raw*.nii.gz",
        t1_thresh=(0, 5000),
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1Molli(
        name="t1_molli_nomdr",
        molli_dir="../fproc/t1_molli_raw_smoothed",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=False,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1SE(
        name="t1_se_nomdr",
        se_dir="../fsort/t1_se_raw",
        tis=np.arange(100, 2001, 100),
        tss=53.7,
        mag_only=True,
    ),
    maps.T2(),
    maps.T2star(),
    maps.B0(),
    maps.B1(),

    # Stitch together potentially multiple slice maps
    regrid.StitchSlices(
        name="t1_molli_stitch",
        img_dir="t1_molli",
        imgs={
            "*t1_map*.nii.g?": "t1_map.nii.gz",  # Just to make sure globs are unique keys!
            "*t1_map*.nii.gz": "t1_conf.nii.gz",
            "*t1_conf*.nii.gz": "t1_conf.nii.gz",
        },
    ),
    regrid.StitchSlices(
        name="t1_molli_nomdr_stitch",
        img_dir="t1_molli_nomdr",
        imgs={
            "*t1_map*.nii.gz": "t1_map.nii.gz",
        },
    ),
    regrid.StitchSlices(
        name="t1_se_nomdr_stitch",
        img_dir="t1_se_nomdr",
        imgs={
            "*t1_map*.nii.gz": "t1_map.nii.gz",
        },
    ),

    # Pre-drawn ROIs
    CopyPreDrawnROIs(),
    
    # Interrogate maps over ROIs
    statistics.SegStats(
        name="stats",
        default_limits=None,
        segs={
            "roi%i" % idx: {
                "dir": "pre_drawn_rois",
                "glob": "voi%i.nii.gz" % idx,
            }
            for idx in range(1, 10)
        },
        params={
            "t1_molli": {
                "dir": "t1_molli_stitch",
                "glob": "t1_conf.nii.gz",
            },
            "t1_molli_nomdr": {
                "dir": "t1_molli_nomdr_stitch",
                "glob": "*map*.nii.gz",
            },
            "t1_se_nomdr": {
                "dir": "t1_se_nomdr_stitch",
                "glob": "*map*.nii.gz",
            },
            "t2_exp": {
                "dir": "t2",
                "glob": "t2_exp.nii.gz",
                "segs": [
                    "kidney_cortex_l",
                    "kidney_cortex_r",
                    "kidney_medulla_l",
                    "kidney_medulla_r",
                ],
            },
            "t2_stim": {
                "dir": "t2",
                "glob": "t2_stim.nii.gz",
                "segs": [
                    "kidney_cortex_l",
                    "kidney_cortex_r",
                    "kidney_medulla_l",
                    "kidney_medulla_r",
                ],
            },
            "b1_stim": {
                "dir": "t2",
                "glob": "b1_stim.nii.gz",
                "segs": [
                    "kidney_cortex_l",
                    "kidney_cortex_r",
                    "kidney_medulla_l",
                    "kidney_medulla_r",
                    "kidney_l",
                    "kidney_r",
                ],
            },
            "t2star_exp": {
                "dir": "t2star",
                "glob": "t2star_2p_exp*.nii.gz",
            },
            "t2star_loglin": {
                "dir": "t2star",
                "glob": "t2star_loglin*.nii.gz",
            },
            "r2star_exp": {
                "dir": "t2star",
                "glob": "r2star_2p_exp*.nii.gz",
            },
            "r2star_loglin": {
                "dir": "t2star",
                "glob": "r2star_loglin*.nii.gz",
            },
            "b0": {
                "dir": "b0",
                "glob": "b0.nii.gz",
            },
            "b1": {
                "dir": "b1",
                "glob": "b1.nii.gz",
            },
        },
        stats=[
            "n",
            "vol",
            "iqn",
            "iqvol",
            "iqmean",
            "median",
            "iqstd",
            "perc90",
            "te",
            "mode",
            "fwhm",
        ],
    )
]
