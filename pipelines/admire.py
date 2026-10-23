# ADMIRE pipeline

from fproc.module import Module
from fproc.modules import (
    maps,
    segmentations,
    statistics,
    seg_postprocess,
    regrid,
    align,
    misc,
)

from afirm import T1DataSmoothed, DixonCorBest, T1MolliMetadata

__version__ = "0.0.1"

# Configuration
NAME = "admire"
STUDYDIR = "/gpfs01/spmstore/project/RenalMRI/admire_pipeline"
OUTNAME = NAME

# Batch submission configuration
BATCH = {
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
        "nnunetv2-img",
        "conda-img",
        "totalseg-img",
        "traceseg-img",
    ],
    "subjects": 10,
    "time": "48:00:00",
    "mem_per_cpu": "64g",
    "extra_fproc_args": [
        "--seg-kidney-fix=/spmstore/project/RenalMRI/admire_pipeline//kidney_masks",
    ],
}

# Output spreadsheet configuration
COHORTS = [
    ("full cohort", "", None),
]

OUTFILES = {
    "kidney": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/stats/stats.csv",
        "fproc/t1_molli_md/t1_molli_md.csv",
    ],
    "t1_radiomics": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/t1_radiomics/radiomics.csv",
    ],
    "organs": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/stats_dixon_totalseg/stats.csv",
        "fproc/kidney_dixon_shape_metrics_totalseg_cor/shape_metrics.csv",
        "fproc/kidney_dixon_radiomics_totalseg_cor/radiomics.csv",
        "fproc/spleen_shape_metrics_totalseg_cor/shape_metrics.csv",
        "fproc/liver_shape_metrics_totalseg_cor/shape_metrics.csv",
        "fproc/pancreas_shape_metrics_totalseg_cor/shape_metrics.csv",
        "fproc/organ_dixon_radiomics/radiomics.csv",
    ],
    "organs_local": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/stats_dixon/stats.csv",
        "fproc/kidney_dixon_shape_metrics_cor/shape_metrics.csv",
        "fproc/kidney_dixon_radiomics_cor/radiomics.csv",
    ],
    "t2w_radiomics": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/t2w_radiomics/radiomics.csv",
    ],
    "shape": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/seg_kidney_t2w_vols/volumes.csv",
        "fproc/tkv_shape_metrics/shape_metrics.csv",
        "fproc/tkv_radiomics/radiomics.csv",
    ],
    "organ_vols": [
        "fproc/scan_dates/scan_dates.csv",
        "fproc/organ_volumes/stats.csv",
    ],
    "totalseg_cor": [
        "fproc/totalseg_cor/volumes.csv",
        "fproc/totalseg_cor/volumes_dilated.csv",
    ],
}


class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats",
            default_limits="3t",
            segs={
                "kidney_cortex": {
                    "dir": "seg_kidney_t1_mdr_align_t2star",
                    "glob": "*cortex*.nii.gz",
                },
                "kidney_medulla": {
                    "dir": "seg_kidney_t1_mdr_align_t2star",
                    "glob": "*medulla*.nii.gz",
                },
                "kidney_t1w": {
                    "dir": "seg_kidney_t1_mdr_align_t2star",
                    "glob": "*.nii.gz",
                    "params": ["b1_stim"],
                },
                "kidney_dixon": {
                    "dir": "totalseg_cor",
                    "glob": "kidneys.nii.gz",
                    "params": ["ff_lt20"],
                },
                "tkv": {
                    "dir": "seg_kidney_t2w_fix",
                    "glob": "kidney_mask.nii.gz",
                },
            },
            params={
                "t2_exp": {
                    "dir": "t2",
                    "glob": "t2_exp.nii.gz",
                    "segs": [
                        "kidney_cortex",
                        "kidney_medulla",
                    ],
                },
                "t2_stim": {
                    "dir": "t2",
                    "glob": "t2_stim.nii.gz",
                    "segs": [
                        "kidney_cortex",
                        "kidney_medulla",
                    ],
                },
                "b1_stim": {
                    "dir": "t2",
                    "glob": "b1_stim.nii.gz",
                    "segs": [
                        "kidney_cortex",
                        "kidney_medulla",
                        "kidney_t1w",
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
                "t1": {
                    "dir": "t1_molli_stitch_fix",
                    "glob": "t1_conf.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex": {"dir": "seg_kidney_t1_fix"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_fix"},
                    },
                },
                "t1_molli_nomdr": {
                    "dir": "t1_molli_nomdr_stitch",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex": {"dir": "seg_kidney_t1_fix"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_fix"},
                    },
                },
                "t1_molli_mdr": {
                    "dir": "t1_molli_mdr_stitch",
                    "glob": "*map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex": {"dir": "seg_kidney_t1_fix"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_fix"},
                    },
                },
                "mtr": {
                    "dir": "mtr",
                    "glob": "mtr.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex": {"dir": "seg_kidney_t1_fix"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_fix"},
                    },
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                    "segs": ["tkv"],
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                    "segs": ["tkv"],
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                    "segs": ["tkv"],
                },
                "adc_mdr": {
                    "dir": "dwi_adc",
                    "glob": "*adc_map*.nii.gz",
                    "seg_overrides": {
                        "kidney_cortex": {"dir": "seg_kidney_t1_fix"},
                        "kidney_medulla": {"dir": "seg_kidney_t1_fix"},
                    },
                },
                "ff_cor": {
                    "dir": "ff_dixon_cor",
                    "glob": "fat_fraction.nii.gz",
                },
                "ff_lt20": {
                    "dir": "ff_dixon_cor",
                    "glob": "fat_fraction.nii.gz",
                    "limits": (0, 20),
                    "segs": ["kidney_dixon"]
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


class StatsDixon(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats_dixon",
            default_limits="3t",
            segs={
                "liver_cor": {
                    "dir": "seg_liver_dixon_cor",
                    "glob": "liver.nii.gz",
                    "seg_volumes": False,
                },
                "spleen_cor": {
                    "dir": "seg_spleen_dixon_cor",
                    "glob": "spleen.nii.gz",
                    "seg_volumes": False,
                },
                "sat_cor": {
                    "dir": "seg_sat_dixon_cor",
                    "glob": "sat.nii.gz",
                    "seg_volumes": False,
                },
                "pancreas": {
                    "dir": "seg_pancreas_ethrive",
                    "glob": "pancreas.nii.gz",
                },
                "kidney_dixon_nofat_cor": {
                    "dir": "seg_kidney_fat_dixon_cor",
                    "glob": "kidney_parenchyma.nii.gz",
                    "params": [],
                },
                "fat_pelvis_cor": {
                    "dir": "seg_kidney_fat_dixon_cor",
                    "glob": "fat_pelvis.nii.gz",
                    "params": ["ff_cor"],
                },
            },
            params={
                "t2star_exp": {
                    "dir": "t2star",
                    "glob": "t2star*_exp.nii.gz",
                },
                "r2star_exp": {
                    "dir": "t2star",
                    "glob": "r2star*_exp.nii.gz",
                },
                "t2star_loglin": {
                    "dir": "t2star",
                    "glob": "t2star*_loglin.nii.gz",
                },
                "r2star_loglin": {
                    "dir": "t2star",
                    "glob": "r2star*_loglin.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_stitch",
                    "glob": "t1_conf.nii.gz",
                },
                "mtr": {
                    "dir": "mtr",
                    "glob": "mtr.nii.gz",
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                },
                "t2_exp": {
                    "dir": "t2",
                    "glob": "t2_exp.nii.gz",
                },
                "t2_stim": {
                    "dir": "t2",
                    "glob": "t2_stim.nii.gz",
                },
                "b1_stim": {
                    "dir": "t2",
                    "glob": "b1_stim.nii.gz",
                },
                "ff_cor": {
                    "dir": "ff_dixon_cor",
                    "glob": "fat_fraction.nii.gz",
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
            seg_volumes=True,
        )


class StatsDixonTotalseg(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self,
            name="stats_dixon_totalseg",
            default_limits="3t",
            segs={
                "liver_cor": {
                    "dir": "totalseg_cor",
                    "glob": "liver.nii.gz",
                    "seg_volumes": False,
                },
                "spleen_cor": {
                    "dir": "totalseg_cor",
                    "glob": "spleen.nii.gz",
                    "seg_volumes": False,
                },
                "sat_cor": {
                    "dir": "totalseg_cor",
                    "glob": "subcutaneous_fat.nii.gz",
                    "seg_volumes": False,
                },
                "pancreas": {
                    "dir": "totalseg_cor",
                    "glob": "pancreas.nii.gz",
                },
                "kidney_dixon_nofat_cor": {
                    "dir": "seg_kidney_fat_dixon_totalseg_cor",
                    "glob": "kidney_parenchyma.nii.gz",
                    "params": [],
                },
                "fat_pelvis_cor": {
                    "dir": "seg_kidney_fat_dixon_totalseg_cor",
                    "glob": "fat_pelvis.nii.gz",
                    "params": ["ff_cor"],
                },
            },
            params={
                "t2star_exp": {
                    "dir": "t2star",
                    "glob": "t2star*_exp.nii.gz",
                },
                "r2star_exp": {
                    "dir": "t2star",
                    "glob": "r2star*_exp.nii.gz",
                },
                "t2star_loglin": {
                    "dir": "t2star",
                    "glob": "t2star*_loglin.nii.gz",
                },
                "r2star_loglin": {
                    "dir": "t2star",
                    "glob": "r2star*_loglin.nii.gz",
                },
                "t1": {
                    "dir": "t1_molli_stitch",
                    "glob": "t1_conf.nii.gz",
                },
                "mtr": {
                    "dir": "mtr",
                    "glob": "mtr.nii.gz",
                },
                "b0": {
                    "dir": "b0",
                    "glob": "b0.nii.gz",
                },
                "b1": {
                    "dir": "b1",
                    "glob": "b1.nii.gz",
                },
                "b1_rescaled": {
                    "dir": "b1",
                    "glob": "b1_rescaled.nii.gz",
                },
                "t2_exp": {
                    "dir": "t2",
                    "glob": "t2_exp.nii.gz",
                },
                "t2_stim": {
                    "dir": "t2",
                    "glob": "t2_stim.nii.gz",
                },
                "b1_stim": {
                    "dir": "t2",
                    "glob": "b1_stim.nii.gz",
                },
                "ff_cor": {
                    "dir": "ff_dixon_cor",
                    "glob": "fat_fraction.nii.gz",
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
            seg_volumes=True,
        )


MODULES = [
    misc.ScanDates(
        "scan_dates",
        input={
            "../fsort/t1w": "*.nii.gz",
            "../fsort/t2w": "*.nii.gz",
            "../fsort/t1_molli": "*.nii.gz",
        },
    ),
    T1MolliMetadata(),
    DixonCorBest(),

    # Parameter maps
    T1DataSmoothed(name="t1_raw_smoothed", dir="../fsort/t1_molli_raw", glob="t1_molli_raw*.nii.gz"),
    T1DataSmoothed(name="t1_maps_smoothed", dir="../fsort/t1_molli", glob="t1_*.nii.gz"),
    maps.T1Molli(
        name="t1_molli",
        molli_dir="../fproc/t1_maps_smoothed",
        molli_glob="t1_molli_raw*.nii.gz",
        t1_thresh=(0, 5000),
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1Molli(
        name="t1_molli_mdr",
        molli_dir="../fproc/t1_raw_smoothed",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=True,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T1Molli(
        name="t1_molli_nomdr",
        molli_dir="../fproc/t1_raw_smoothed",
        molli_glob="t1_molli_raw*.nii.gz",
        mdr=False,
        use_scanner_maps=False,
        tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0],
        tis_use_md=True,
    ),
    maps.T2(),
    maps.T2star(),
    maps.MTR(),
    maps.B0(),
    maps.B1(),
    maps.FatFractionDixon(name="ff_dixon_cor", dixon_dir="../fproc/dixon_cor_best"),
    maps.DwiMoco(),
    maps.DwiAdc(),

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
        name="t1_molli_mdr_stitch",
        img_dir="t1_molli_mdr",
        imgs={
            "*t1_map*.nii.gz": "t1_map.nii.gz",
        },
    ),

    # Segmentations
    segmentations.SatDixon(
        name="seg_sat_dixon_cor", dixon_dir="../fproc/dixon_cor_best"
    ),
    segmentations.LiverDixon(
        name="seg_liver_dixon_cor", dixon_dir="../fproc/dixon_cor_best"
    ),
    segmentations.SpleenDixon(
        name="seg_spleen_dixon_cor", dixon_dir="../fproc/dixon_cor_best"
    ),
    segmentations.KidneyDixon(
        name="seg_kidney_dixon_cor", dixon_dir="../fproc/dixon_cor_best", model_id="422"
    ),
    segmentations.PancreasEthrive(),
    segmentations.BodyDixon(
        name="seg_body_dixon_cor", dixon_dir="../fproc/dixon_cor_best"
    ),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive", "pancreas.nii.gz"),
    segmentations.TotalSeg(
        name="totalseg_cor",
        src_dir="../fproc/dixon_cor_best",
        dilate=1,
        csv_suffix="_cor",
    ),
    segmentations.VatDixon(
        name="seg_vat_dixon_cor_local",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        body_dir="seg_body_dixon_cor",
        sat_dir="seg_sat_dixon_cor",
        organs={
            "seg_liver_dixon_cor": "liver.nii.gz",
            "seg_spleen_dixon_cor": "spleen.nii.gz",
            "seg_pancreas_ethrive_largestblob": "pancreas.nii.gz",
            "seg_kidney_dixon_cor": "kidney.nii.gz",
        },
    ),
    segmentations.TotalSeg(
        name="totalseg_sag_loc",
        src_dir="sag_local",
        water_glob="sag_local.nii.gz",
        fat_glob=None,
        dilate=1,
        csv_suffix="_sagloc",
    ),
    segmentations.VatDixon(
        name="seg_vat_dixon_cor_totalseg",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        body_dir="seg_body_dixon_cor",
        sat_dir="totalseg_cor",
        sat_glob="subcutaneous_fat.nii.gz",
        dixon_dir="../fproc/dixon_cor_best",
        organs={
            "totalseg_cor": "liver.nii.gz",
            "totalseg_cor": "spleen.nii.gz",
            "totalseg_cor": "pancreas.nii.gz",
            "totalseg_cor": "kidneys.nii.gz",
        },
    ),
    regrid.CombineSegs(
        name="seg_liver_combined",
        seg_dirs=["../fproc/totalseg_cor", "../fproc/totalseg_sag_loc"],
        seg_globs=["liver.nii.gz", "liver.nii.gz"],
        out_fname="liver.nii.gz",
    ),
    regrid.CombineSegs(
        name="seg_pancreas_combined",
        seg_dirs=["../fproc/totalseg_cor", "../fproc/totalseg_sag_loc"],
        seg_globs=["pancreas.nii.gz", "pancreas.nii.gz"],
        out_fname="pancreas.nii.gz",
    ),
    regrid.CombineSegs(
        name="seg_spleen_combined",
        seg_dirs=["../fproc/totalseg_cor", "../fproc/totalseg_sag_loc"],
        seg_globs=["spleen.nii.gz", "spleen.nii.gz"],
        out_fname="spleen.nii.gz",
    ),
    segmentations.TraceSeg(name="traceseg", src_dir="t2w", img_glob="t2w.nii.gz"),

    # Manual masks and fixes
    seg_postprocess.SegFix(
        "seg_kidney_t1",
        fix_dir_option="seg_kidney_fix",
        segs={
            "kidney_cortex.nii.gz": {
                "glob": "%s/kidney_combined_tissues.nii.gz",
                "fname": "kidney_cortex.nii.gz",
                "label": 1
            },
            "kidney_medulla.nii.gz": {
                "glob": "%s/kidney_combined_tissues.nii.gz",
                "fname": "kidney_medulla.nii.gz",
                "label": 2
            },
        },
        map_dir="t1_molli_stitch",
        map_fname="t1_map.nii.gz",
    ),
    seg_postprocess.SegFix(
        "seg_kidney_t2w",
        fix_dir_option="seg_kidney_fix",
        segs={
            "kidney_mask.nii.gz": {
                "fname": "kidney_mask.nii.gz",
                "glob": "%s/kidney_mask.nii.gz",
            },
        },
        map_dir="t2w",
        map_fname="t2w.nii.gz",
        map_src=Module.INPUT,
    ),

    # Re-alignments
    align.FlirtAlignOnly(
        name="seg_kidney_t1_align_t2star",
        in_dir="t1_molli_stitch_fix",
        in_glob="t1_map.nii.gz",
        ref_dir="t2star",
        ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w_fix",
        weight_mask="kidney_mask.nii.gz",
        weight_mask_dil=6,
        also_align={
            "seg_kidney_t1_fix": "*.nii.gz",
            "t1_molli_stitch_fix": "t1_conf.nii.gz",
        },
    ),
    align.FlirtAlignOnly(
        name="seg_kidney_t1_mdr_align_t2star",
        in_dir="t1_molli_mdr_stitch",
        in_glob="t1_map.nii.gz",
        ref_dir="t2star",
        ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w_fix",
        weight_mask="kidney_mask.nii.gz",
        weight_mask_dil=6,
        also_align={
            "seg_kidney_t1_fix": "kidney*.nii.gz",
            "t1_molli_mdr_stitch": "t1_conf.nii.gz",
        },
    ),

    # Derived segmentations
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_cor",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        kidney_seg_dir="seg_kidney_dixon_cor",
        kidney_seg_glob="kidney.nii.gz",
        ff_thresh=15,
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_totalseg_cor",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction.nii.gz",
        kidney_seg_dir="totalseg_cor",
        kidney_seg_glob="kidneys.nii.gz",
        ff_thresh=15,
    ),

    # Statistics and numerical measures
    seg_postprocess.SegVolumes(
        "seg_kidney_t2w_vols",
        seg_dir="seg_kidney_t2w_fix",
        segs={
            "kv_mask": "*mask*.nii.gz",
        },
    ),
    statistics.Radiomics(
        name="t2w_radiomics",
        deps=["seg_kidney_t2w_fix"],
        params={
            "t2w": {"dir": "../fsort/t2w", "fname": "t2w.nii.gz"},
        },
        segs={
            "tkv": {"dir": "seg_kidney_t2w_fix", "fname": "kidney_mask.nii.gz"},
        },
        image_types=["Original", "Exponential"],
        features=["firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"],
    ),
    Stats(),
    StatsDixon(),
    StatsDixonTotalseg(),
    statistics.CMD(
        deps=["stats"],
        cmd_params=["t2star", "t2star", "t1", "mtr"],
    ),
    statistics.ShapeMetrics(
        name="tkv_shape_metrics",
        seg_dir="seg_kidney_t2w_fix",
        segs={"tkv": "kidney_mask.nii.gz"},
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
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_cor",
        seg_dir="seg_kidney_dixon_cor",
        segs={
            "kidney_cor": "kidney.nii.gz",
        },
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
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_totalseg_cor",
        seg_dir="totalseg_cor",
        segs={
            "kidney_cor": "kidney.nii.gz",
        },
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
    statistics.ShapeMetrics(
        name="spleen_shape_metrics_totalseg_cor",
        seg_dir="seg_spleen_combined",
        segs={
            "spleen_comb": "spleen.nii.gz",
        },
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
    statistics.ShapeMetrics(
        name="liver_shape_metrics_totalseg_cor",
        seg_dir="seg_liver_combined",
        segs={
            "liver_comb": "liver.nii.gz",
        },
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
    statistics.ShapeMetrics(
        name="pancreas_shape_metrics_totalseg_cor",
        seg_dir="seg_pancreas_combined",
        segs={
            "pancreas_comb": "pancreas.nii.gz",
        },
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
    statistics.Radiomics(
        name="tkv_radiomics",
        deps=["seg_kidney_t2w_fix"],
        params={
            "t2w": {"dir": "t2w", "fname": "t2w.nii.gz", "src": Module.INPUT},
        },
        segs={
            "tkv": {"dir": "seg_kidney_t2w_fix", "fname": "kidney_mask.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="kidney_dixon_radiomics_cor",
        deps=["t1_molli_stitch_fix", "seg_kidney_dixon_cor"],
        params={
            "t1": {
                "dir": "t1_molli_stitch_fix",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_cor": {
                "dir": "seg_kidney_dixon_cor",
                "fname": "*.nii.gz",
            },
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="kidney_dixon_radiomics_totalseg_cor",
        deps=["t1_molli_stitch_fix", "totalseg_cor"],
        params={
            "t1": {
                "dir": "t1_molli_stitch_fix",
                "fname": "t1_map.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "kidney_dixon_cor": {
                "dir": "totalseg_cor",
                "fname": "*kidney*.nii.gz",
            },
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="organ_dixon_radiomics",
        deps=[
            "totalseg_cor",
            "seg_liver_combined",
            "seg_pancreas_combined",
            "seg_spleen_combined",
        ],
        params={
            "water": {
                "dir": "dixon_cor_best",
                "fname": "water.nii.gz",
                "src": Module.OUTPUT,
            },
        },
        segs={
            "liver": {
                "dir": "seg_liver_combined",
                "fname": "liver.nii.gz",
            },
            "pancreas": {
                "dir": "seg_pancreas_combined",
                "fname": "pancreas.nii.gz",
            },
            "spleen": {
                "dir": "seg_spleen_combined",
                "fname": "spleen.nii.gz",
            },
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ),
    statistics.Radiomics(
        name="t1_radiomics",
        deps=["t1_scaled", "t2", "seg_kidney_t1_clean_native", "t1_molli_mdr_stitch", "seg_kidney_t1_mdr_clean_native_proc"],
        params={
            "t1_mdr": {"dir": "t1_molli_mdr_stitch", "fname": "t1_map.nii.gz"},
            "t2_exp": {"dir": "t2", "fname": "t2_exp.nii.gz"},
            "t2_stim": {"dir": "t2", "fname": "t2_stim.nii.gz"},
        },
        segs={
            "cortex": {
                "dir": "seg_kidney_t1_fix",
                "fname": "*cortex*.nii.gz",
            },
            "medulla": {
                "dir": "seg_kidney_t1_fix",
                "fname": "*medulla*.nii.gz",
            },
        },
    ),
    statistics.ISNR(
        src=Module.INPUT,
        imgs={
            "t1w": "t1w.nii.gz",
            "t2w": "t2w.nii.gz",
        },
    ),
    statistics.SegStats(
        name="organ_volumes",
        segs={
            "liver_totalseg_cor": {
                "dir": "totalseg_cor",
                "glob": "liver.nii.gz",
            },
            "liver_comb": {
                "dir": "seg_liver_combined",
                "glob": "liver.nii.gz",
            },
            "pancreas_local": {
                "dir": "seg_pancreas_ethrive",
                "glob": "pancreas.nii.gz",
            },
            "pancreas_totalseg_cor": {
                "dir": "totalseg_cor",
                "glob": "pancreas.nii.gz",
            },
            "pancreas_comb": {
                "dir": "seg_pancreas_combined",
                "glob": "pancreas.nii.gz",
            },
            "spleen_totalseg_cor": {
                "dir": "totalseg_cor",
                "glob": "spleen.nii.gz",
            },
            "sat_totalseg_cor": {
                "dir": "totalseg_cor",
                "glob": "subcutaneous_fat.nii.gz",
            },
            "vat_cor_local": {
                "dir": "seg_vat_dixon_cor_local",
                "glob": "vat.nii.gz",
            },
            "vat_cor_totalseg": {
                "dir": "seg_vat_dixon_cor_totalseg",
                "glob": "vat.nii.gz",
            },
        },
        seg_volumes=True,
    ),
]


def add_options(parser):
    parser.add_argument(
        "--seg-kidney-fix", help="Directory containing manual kidney masks"
    )
