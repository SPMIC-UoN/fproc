import logging

import numpy as np

from fproc.module import Module
from fproc.modules import maps, segmentations, statistics, seg_postprocess, regrid, align

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME="afirm_t2"

class StatsRenalPreproc(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats_renal_preproc",
            default_limits="3t",
            segs = {
                "kidney_cortex" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*cortex_t1*.nii.gz",
                },
                "kidney_cortex_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*cortex_l*.nii.gz",
                },
                "kidney_cortex_r" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*cortex_r*.nii.gz",
                },
                "kidney_medulla" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*medulla_t1*.nii.gz",
                },
                "kidney_medulla_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*medulla_l*.nii.gz",
                },
                "kidney_medulla_r" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*medulla_r*.nii.gz",
                },
                "tkv_l" : {
                    "dir" : "seg_kidney_t2w_fix",
                    "glob" : "*left*.nii.gz",
                },
                "tkv_r" : {
                    "dir" : "seg_kidney_t2w_fix",
                    "glob" : "*right*.nii.gz",
                },
            },
            params = {
                "t2star_exp" : {
                    "dir" : "t2star",
                    "glob" : "t2star_2p_exp*.nii.gz",
                },
                "t2star_loglin" : {
                    "dir" : "t2star",
                    "glob" : "t2star_loglin*.nii.gz",
                },
                "r2star_exp" : {
                    "dir" : "t2star",
                    "glob" : "r2star_2p_exp*.nii.gz",
                },
                "r2star_loglin" : {
                    "dir" : "t2star",
                    "glob" : "r2star_loglin*.nii.gz",
                },
                "t1" : {
                    "dir" : "t1_stitch_molli",
                    "glob" : "*conf*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_clean_native"},
                    }
                },
                "t1_noclean" : {
                    "dir" : "t1_stitch_molli",
                    "glob" : "*conf*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_clean_native_generic"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_clean_native_generic"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_clean_native_generic"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_clean_native_generic"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_clean_native_generic"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_clean_native_generic"},
                    }
                },
                "t1_molli_nomdr" : {
                    "dir" : "t1_stitch_molli_nomdr",
                    "glob" : "*map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_clean_native"},
                    }
                },
                "t1_molli_mdr" : {
                    "dir" : "t1_stitch_molli_mdr",
                    "glob" : "*map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_clean_native"},
                    }
                },
                "t1_se_nomdr" : {
                    "dir" : "t1_stitch_se_nomdr",
                    "glob" : "*map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_clean_native"},
                    }
                },
                "t1_se_mdr" : {
                    "dir" : "t1_stitch_se_mdr",
                    "glob" : "*map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_clean_native"},
                    }
                },
                 "mtr" : {
                    "dir" : "mtr",
                    "glob" : "mtr.nii.gz",
                },
                "b0" : {
                    "dir" : "b0",
                    "glob" : "b0.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"],
                },
                "b1" : {
                    "dir" : "b1",
                    "glob" : "b1.nii.gz",
                    "segs" : ["tkv_l", "tkv_r"],
                },
                "b1_rescaled" : {
                    "dir" : "b1",
                    "glob" : "b1_rescaled.nii.gz",
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
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*cortex_r*.nii.gz"
                },
                "cortex_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*cortex_l*.nii.gz"
                },
                "medulla_r" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*medulla_r*.nii.gz"
                },
                "medulla_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*medulla_l*.nii.gz"
                },
                "kidney_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*all_l*.nii.gz"
                },
                "kidney_r" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*all_r*.nii.gz"
                },
                "tkv_l" : {
                    "dir" : "seg_kidney_t2w_fix",
                    "glob" : "*left*.nii.gz"
                },
                "tkv_r" : {
                    "dir" : "seg_kidney_t2w_fix",
                    "glob" : "*right*.nii.gz"
                },
            },
            params={
                "t2_exp" : {
                    "dir" : "t2",
                    "glob" : "t2_exp.nii.gz",
                    "segs" : ["kidney_cortex_l", "kidney_cortex_r", "kidney_medulla_l", "kidney_medulla_r"],
                },
                "t2_stim" : {
                    "dir" : "t2",
                    "glob" : "t2_stim.nii.gz",
                    "segs" : ["kidney_cortex_l", "kidney_cortex_r", "kidney_medulla_l", "kidney_medulla_r"],
                },
                "b1_stim" : {
                    "dir" : "t2",
                    "glob" : "b1_stim.nii.gz",
                    "segs" : ["kidney_cortex_l", "kidney_cortex_r", "kidney_medulla_l", "kidney_medulla_r", "kidney_l", "kidney_r"],
                },
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
                    "dir" : "t2star",
                    "glob" : "t2star*_exp.nii.gz",
                },
                "r2star_exp" : {
                    "dir" : "t2star",
                    "glob" : "r2star*_exp.nii.gz",
                },
                "t2star_loglin" : {
                    "dir" : "t2star",
                    "glob" : "t2star*_loglin.nii.gz",
                },
                "r2star_loglin" : {
                    "dir" : "t2star",
                    "glob" : "r2star*_loglin.nii.gz",
                },
                "t1" : {
                    "dir" : "t1_stitch_molli",
                    "glob" : "map_t1_conf.nii.gz",
                },
                "t1_ax" : {
                    "dir" : "molli_ax",
                    "src" : self.INPUT,
                    "glob" : "t1_conf.nii.gz",
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
                    "glob" : "b1_noscale.nii.gz",
                },
                "b1_rescaled" : {
                    "dir" : "b1",
                    "glob" : "b1_rescaled.nii.gz",
                },
                "t2_exp" : {
                    "dir" : "t2",
                    "glob" : "t2_exp.nii.gz",
                },
                "t2_stim" : {
                    "dir" : "t2",
                    "glob" : "t2_stim.nii.gz",
                },
                "b1_stim" : {
                    "dir" : "t2",
                    "glob" : "b1_stim.nii.gz",
                },
                "ff_cor" : {
                    "dir" : "ff_dixon_cor",
                    "glob" : "fat_fraction_scanner.nii.gz",
                },
                "ff_ax" : {
                    "dir" : "ff_dixon_ax",
                    "glob" : "fat_fraction_scanner.nii.gz",
                },
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
            seg_volumes=False,
        )

class TKVRadiomics(statistics.Radiomics):
    def __init__(self):
        statistics.Radiomics.__init__(
            self,
            name="tkv_shape_metrics",
            out_name="tkv_shape_metrics.csv",
            params={
                "t2w" : {"dir" : "t2w", "fname" : "t2w.nii.gz", "src" : Module.INPUT},
            },
            segs={
                "tkv_l" : {"dir" : "seg_kidney_t2w_fix", "fname" : "*left*.nii.gz"},
                "tkv_r" : {"dir" : "seg_kidney_t2w_fix", "fname" : "*right*.nii.gz"},
            },
            features=[
                "shape"
            ],
        )

class T1MolliMetadata(Module):
    def __init__(self, name="t1_molli_md", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t1_dir = "t1_molli"
        t1_glob = "t1_*.nii.gz"
        t1s = self.inimgs(t1_dir, t1_glob, src=self.INPUT)
        if not t1s:
            self.no_data(f"No T1 maps found in {t1_dir} matching {t1_glob}")

        tis = []
        hr = []
        for t1 in t1s:
            tis.extend(list(t1.inversiontimedelay))
            hr.extend(list(t1.heartrate))

        hr = np.unique(hr)
        if len(hr) > 1:
            LOG.warn(f"Multiple heart rates found: {hr} - using first")
            hr = hr[0]
        elif len(hr) == 0:
            LOG.warn("No heart rate found")
            hr = ""            
        else:
            hr = hr[0]
            LOG.info(f" - Found heart rate: {hr}")

        tis = sorted([float(v) for v in np.unique(tis) if float(v) > 0])
        with open(self.outfile("tis.txt"), "w") as f:
            f.write("\n".join([str(v) for v in tis]))

        LOG.info(f" - Found TIs: {tis}")
        if len(tis) >= 3:
            ti1, ti2, spacing = tis[0], tis[1], tis[2] - tis[0]
        else:
            ti1, ti2, spacing = "", "", ""
            LOG.warn(f"Not enough TIs found: {tis}")
        
        with open(self.outfile("t1_molli_md.csv"), "w") as f:
            f.write(f"t1_molli_heart_rate,{hr}\n")
            f.write(f"t1_molli_ti1,{ti1}\n")
            f.write(f"t1_molli_ti2,{ti2}\n")
            f.write(f"t1_molli_ti_spacing,{spacing}\n")

class T1Scaled(Module):

    def __init__(self, name="t1_scaled", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        scale_factors_fname = self.pipeline.options.t1_scale_factors
        sf_left, sf_right = None, None
        with open(scale_factors_fname) as f:
            for l in f.readlines()[1:]:
                parts = l.split(",")
                if len(parts) != 3:
                    LOG.warn(f"Invalid line in scale factors file: {l}")
                    continue
                subjid = parts[0]
                if subjid.strip().lower() == self.pipeline.options.subjid.strip().lower():
                    sf_left, sf_right = float(parts[1]), float(parts[2])
                    break
        if not sf_left and not sf_right:
            LOG.warn(f"Scale factors not found for {self.pipeline.options.subjid}")
            return

        t1_dir = "t1_stitch_molli"
        t1_glob = "*map_t1*.nii.gz"
        t1s = self.inimgs(t1_dir, t1_glob, src=self.OUTPUT)
        if not t1s:
            self.no_data(f"No T1 maps found in {t1_dir} matching {t1_glob}")

        for t1 in t1s:
            left_data = t1.data * sf_left
            right_data = t1.data * sf_right
            t1.save_derived(left_data, self.outfile(t1.fname.replace(".nii.gz", "_scaled_left.nii.gz")))
            t1.save_derived(right_data, self.outfile(t1.fname.replace(".nii.gz", "_scaled_right.nii.gz")))

class T1RadiomicsLeft(statistics.Radiomics):
    def __init__(self):
        statistics.Radiomics.__init__(
            self,
            name="t1_radiomics_left",
            params={
                "t1" : {"dir" : "t1_scaled", "fname" : "map_t1_conf_scaled_left.nii.gz"},
                "t2_exp" : {"dir" : "t2", "fname" : "t2_exp.nii.gz"},
                "t2_stim" : {"dir" : "t2", "fname" : "t2_stim.nii.gz"},
            },
            segs={
                "cortex_l" : {"dir" : "seg_kidney_t1", "fname" : "kidney_cortex_l_t1.nii.gz"},
                "medulla_l" : {"dir" : "seg_kidney_t1", "fname" : "kidney_medulla_l_t1.nii.gz"},
            },
        )

class T1RadiomicsRight(statistics.Radiomics):
    def __init__(self):
        statistics.Radiomics.__init__(
            self,
            name="t1_radiomics_right",
            params={
                "t1" : {"dir" : "t1_scaled", "fname" : "map_t1_conf_scaled_right.nii.gz"},
                "t2_exp" : {"dir" : "t2", "fname" : "t2_exp.nii.gz"},
                "t2_stim" : {"dir" : "t2", "fname" : "t2_stim.nii.gz"},
            },
            segs={
                "cortex_r" : {"dir" : "seg_kidney_t1", "fname" : "kidney_cortex_r_t1.nii.gz"},
                "medulla_r" : {"dir" : "seg_kidney_t1", "fname" : "kidney_medulla_r_t1.nii.gz"},
            },
        )

MODULES = [
    # Derived maps
    maps.T1Molli(name="t1_molli_nocorr", molli_dir="t1_molli", molli_glob="t1_raw_molli*.nii.gz", molli=False, parameters=2),
    maps.T1Molli(name="t1_molli", molli_dir="t1_molli", molli_glob="t1_raw_molli*.nii.gz", t1_thresh=(0, 5000)),
    maps.T1Molli(name="t1_molli_mdr", molli_dir="t1_molli_raw", molli_glob="t1_molli_raw*.nii.gz", mdr=True, use_scanner_maps=False),
    maps.T1Molli(name="t1_molli_nomdr", molli_dir="t1_molli_raw", molli_glob="t1_molli_raw*.nii.gz", mdr=False, use_scanner_maps=False),
    maps.T1SE(se_dir="t1_se_raw", tis=np.arange(100, 2001, 100), tss=53.7, name="t1_se_nomdr"),
    maps.T1SE(se_dir="t1_se_raw", tis=np.arange(100, 2001, 100), tss=53.7, name="t1_se_mdr", mdr=True),
    regrid.StitchSlices(
        "t1_stitch_molli",
        img_dir="t1_molli",
        imgs={
            "*map_t1*.nii.gz" : "map_t1.nii.gz",
            "*t1_map*.nii.gz" : "map_t1.nii.gz",
            "*conf_*.nii.gz" : "map_t1_conf.nii.gz",
            "*_t1_map*.nii.gz" : "map_t1_conf.nii.gz",
            "*t1_conf*.nii.gz" : "map_t1_conf.nii.gz",
        }
    ), 
    regrid.StitchSlices(
        "t1_stitch_molli_nocorr",
        img_dir="t1_molli_nocorr",
        imgs={
            "*map_t1*.nii.gz" : "map_t1.nii.gz",
            "*t1_map*.nii.gz" : "map_t1.nii.gz",
            "*conf_*.nii.gz" : "map_t1_conf.nii.gz",
            "*_t1_map*.nii.gz" : "map_t1_conf.nii.gz",
            "*t1_conf*.nii.gz" : "map_t1_conf.nii.gz",
        }
    ),
    regrid.StitchSlices(
        "t1_stitch_molli_nomdr",
        img_dir="t1_molli_nomdr",
        imgs={
            "*t1_map*.nii.gz" : "map_t1.nii.gz",
        }
    ),
    regrid.StitchSlices(
        "t1_stitch_molli_mdr",
        img_dir="t1_molli_mdr",
        imgs={
            "*t1_map*.nii.gz" : "map_t1.nii.gz",
        }
    ),
    regrid.StitchSlices(
        "t1_stitch_se_nomdr",
        img_dir="t1_se_nomdr",
        imgs={
            "*t1_map*.nii.gz" : "map_t1.nii.gz",
        }
    ),
    regrid.StitchSlices(
        "t1_stitch_se_mdr",
        img_dir="t1_se_mdr",
        imgs={
            "*t1_map*.nii.gz" : "map_t1.nii.gz",
        }
    ),
    maps.T2(),
    maps.T2star(),
    maps.FatFractionDixon(name="ff_dixon_cor", dixon_dir="dixon_cor"),
    maps.FatFractionDixon(name="ff_dixon_ax", dixon_dir="dixon_ax"),
    maps.MTR(),
    maps.B0(),
    maps.B1(),

    # Segmentations
    segmentations.KidneyCystT2w(t2w_dir="t2w", t2w_glob="t2w.nii.gz", t2w_src=Module.INPUT),
    segmentations.KidneyT1(map_dir="t1_stitch_molli", map_glob="map_t1_conf.nii.gz"),
    segmentations.KidneyT1(name="seg_kidney_t1_nocorr", map_dir="t1_stitch_molli_nocorr", map_glob="map_t1_conf.nii.gz"),
    segmentations.KidneyT1(name="seg_kidney_t1_mdr", map_dir="t1_stitch_molli_mdr", map_glob="map_t1.nii.gz"),
    segmentations.KidneyT2w(),
    #segmentations.BodyDixon(),
    segmentations.SatDixon(name="seg_sat_dixon_cor", dixon_dir="dixon_cor"),
    segmentations.LiverDixon(name="seg_liver_dixon_cor", dixon_dir="dixon_cor"),
    segmentations.SpleenDixon(name="seg_spleen_dixon_cor",dixon_dir="dixon_cor"),
    segmentations.SatDixon(name="seg_sat_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.LiverDixon(name="seg_liver_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.SpleenDixon(name="seg_spleen_dixon_ax", dixon_dir="dixon_ax"),
    #segmentations.KidneyDixon(model_id="422"),
    segmentations.PancreasEthrive(),

    # Post-processing of segmentations
    seg_postprocess.SegFix(
        "seg_kidney_cyst_t2w",
        fix_dir_option="cyst_masks",
        segs={
            "kidney_cyst_mask.nii.gz" : "%s/*FIX*.nii.gz",
        },
        map_dir="t2w",
        map_fname="t2w.nii.gz",
        map_src=Module.INPUT,
    ),
    seg_postprocess.SegFix(
        "seg_kidney_t2w",
        fix_dir_option="seg_kidney_t2w_fix",
        segs={
            "*mask*" : {
                "fname" : "kidney_mask.nii.gz",
                "glob" : "%s/**/*FIX*.nii.gz",
            },
            "*left*" : {
                "fname" : "kidney_left.nii.gz",
                "glob" : "%s/**/*FIX*.nii.gz",
                "side" : "left",
            },
            "*right*" : {
                "fname" : "kidney_right.nii.gz",
                "glob" : "%s/**/*FIX*.nii.gz",
                "side" : "right",
            },
        },
        map_dir="t2w",
        map_fname="t2w.nii.gz",
        map_src=Module.INPUT,
    ),
    align.FlirtAlignOnly(
        name="seg_kidney_t1_align_t2star",
        in_dir="t1_stitch_molli",
        in_glob="map_t1.nii.gz",
        ref_dir="t2star",
        ref_glob="last_echo.nii.gz",
        weight_mask_dir="seg_kidney_t2w_fix",
        weight_mask="kidney_mask.nii.gz",
        weight_mask_dil=6,
        also_align={
            "seg_kidney_t1" : "kidney*.nii.gz",
        }
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean",
        srcdir="seg_kidney_t1_align_t2star",
        seg_t1_glob="kidney_*.nii.gz",
        t1_map_srcdir="seg_kidney_t1_align_t2star",
        t1_map_glob="map_t1.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean_native",
        srcdir="seg_kidney_t1",
        seg_t1_glob="kidney_*.nii.gz",
        t1_map_srcdir="t1_stitch_molli",
        t1_map_glob="map_t1.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean_native_generic",
        srcdir="seg_kidney_t1",
        seg_t1_glob="kidney_*.nii.gz",
        t1_map_srcdir="t1_stitch_molli",
        t1_map_glob="map_t1.nii.gz",
        t2w=False
    ),
    seg_postprocess.KidneyCystClean(
        cyst_dir="seg_kidney_cyst_t2w_fix",
        seg_t2w_dir="seg_kidney_t2w_fix",
        t2w_dir="t2w",
        t2w_glob="t2w.nii.gz",
        t2w_src=Module.INPUT,
    ),

    # Statistics outputs
    Stats(),
    StatsDixon(),
    StatsRenalPreproc(),
    T1MolliMetadata(),
    TKVRadiomics(),
    T1Scaled(),
    T1RadiomicsLeft(),
    T1RadiomicsRight(),
]

def add_options(parser):
    parser.add_argument("--t1-scale-factors", help="Directory containing manual fixed cyst masks")
    parser.add_argument("--cyst-masks", help="Directory containing manual fixed cyst masks")
    parser.add_argument("--seg-kidney-t2w-fix", help="Directory containing manual T2w kidney masks")
