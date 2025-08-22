import logging
import os

import numpy as np

from fproc.module import Module
from fproc.modules import maps, segmentations, statistics, seg_postprocess, regrid, align

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME="afirm"

class Stats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats",
            default_limits="3t",
            segs = {
                "kidney_cortex" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*cortex*.nii.gz",
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
                    "glob" : "*medulla*.nii.gz",
                },
                "kidney_medulla_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*medulla_l*.nii.gz",
                },
                "kidney_medulla_r" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*medulla_r*.nii.gz",
                },
                "kidney_l" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*_l.nii.gz",
                    "params" : ["b1_stim"]
                },
                "kidney_r" : {
                    "dir" : "seg_kidney_t1_clean",
                    "glob" : "*_r.nii.gz",
                    "params" : ["b1_stim"]
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
                    "dir" : "t1_molli_stitch_fix",
                    "glob" : "t1_conf.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_clean_native"},
                    }
                },
                "t1_molli_nomdr" : {
                    "dir" : "t1_molli_nomdr_stitch",
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
                    "dir" : "t1_molli_mdr_stitch",
                    "glob" : "*map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                    }
                },
                "t1_se_nomdr" : {
                    "dir" : "t1_se_nomdr_stitch",
                    "glob" : "*map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_se_clean_native"},
                    }
                },
                "t1_se_mdr_2p" : {
                    "dir" : "t1_se_mdr_stitch",
                    "glob" : "*map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_se_clean_native"},
                    }
                },
                "t1_se_mdr_3p" : {
                    "dir" : "t1_se_mdr_step2_stitch",
                    "glob" : "*map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_se_clean_native"},
                    }
                },
                 "mtr" : {
                    "dir" : "mtr",
                    "glob" : "mtr.nii.gz",
                },
                 "mtr_mdr" : {
                    "dir" : "mtr",
                    "glob" : "mtr.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_mdr_clean_native"},
                    }
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
                "adc_mdr" : {
                    "dir" : "dwi_adc",
                    "glob" : "*adc_map*.nii.gz",
                    "seg_overrides" : {
                        "kidney_cortex_l" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex_r" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_cortex" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_l" : {"dir" : "seg_kidney_t1_se_clean_native"},
                        "kidney_medulla_r" : {"dir" : "seg_kidney_t1_se_clean_native"},
                    }
                },
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
        )

class StatsDixon(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats_dixon", 
            default_limits="3t",
            segs={
                 "liver_cor" : {
                    "dir" : "seg_liver_dixon_cor",
                    "glob" : "liver.nii.gz",
                    "seg_volumes" : False,
                },
                "spleen_cor" : {
                    "dir" : "seg_spleen_dixon_cor",
                    "glob" : "spleen.nii.gz",
                    "seg_volumes" : False,
                },
                "sat_cor" : {
                    "dir" : "seg_sat_dixon_cor",
                    "glob" : "sat.nii.gz",
                    "seg_volumes" : False,
                },
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
                "vat_ax" : {
                    "dir" : "seg_vat_dixon_ax",
                    "glob" : "vat.nii.gz",
                    "params" : []  # Volumes only
                },
                "pancreas" : {
                    "dir" : "seg_pancreas_ethrive",
                    "glob" : "pancreas.nii.gz",
                },
                "kidney_dixon_nofat_cor" : {
                    "dir" : "seg_kidney_fat_dixon_cor",
                    "glob" : "kidney_parenchyma.nii.gz",
                    "params" : [],
                },
                "kidney_dixon_left_nofat_cor" : {
                    "dir" : "seg_kidney_fat_dixon_cor",
                    "glob" : "kidney_parenchyma_left.nii.gz",
                    "params" : [],
                },
                "kidney_dixon_right_nofat_cor" : {
                    "dir" : "seg_kidney_fat_dixon_cor",
                    "glob" : "kidney_parenchyma_right.nii.gz",
                    "params" : [],
                },
                "kidney_dixon_nofat_ax" : {
                    "dir" : "seg_kidney_fat_dixon_ax",
                    "glob" : "kidney_parenchyma.nii.gz",
                    "params" : [],
                },
                "kidney_dixon_left_nofat_ax" : {
                    "dir" : "seg_kidney_fat_dixon_ax",
                    "glob" : "kidney_parenchyma_left.nii.gz",
                    "params" : [],
                },
                "kidney_dixon_right_nofat_ax" : {
                    "dir" : "seg_kidney_fat_dixon_ax",
                    "glob" : "kidney_parenchyma_right.nii.gz",
                    "params" : [],
                },
                "fat_pelvis_cor" : {
                    "dir" : "seg_kidney_fat_dixon_cor",
                    "glob" : "fat_pelvis.nii.gz",
                    "params" : ["ff_cor"],
                },
                "fat_pelvis_left_cor" : {
                    "dir" : "seg_kidney_fat_dixon_cor",
                    "glob" : "fat_pelvis_left.nii.gz",
                    "params" : ["ff_cor"],
                },
                "fat_pelvis_right_cor" : {
                    "dir" : "seg_kidney_fat_dixon_cor",
                    "glob" : "fat_pelvis_right.nii.gz",
                    "params" : ["ff_cor"],
                },
                "fat_pelvis_ax" : {
                    "dir" : "seg_kidney_fat_dixon_ax",
                    "glob" : "fat_pelvis.nii.gz",
                    "params" : ["ff_ax"],
                },
                "fat_pelvis_left_ax" : {
                    "dir" : "seg_kidney_fat_dixon_ax",
                    "glob" : "fat_pelvis_left.nii.gz",
                    "params" : ["ff_ax"],
                },
                "fat_pelvis_right_ax" : {
                    "dir" : "seg_kidney_fat_dixon_ax",
                    "glob" : "fat_pelvis_right.nii.gz",
                    "params" : ["ff_ax"],
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
                    "dir" : "t1_molli_stitch",
                    "glob" : "t1_conf.nii.gz",
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
            seg_volumes=True,
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
        if scale_factors_fname and os.path.isfile(scale_factors_fname):
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

        if not sf_left:
            LOG.warn(f"Scale factor (left) not found for {self.pipeline.options.subjid} - will output unscaled data")
            sf_left = 1
        if not sf_right:
            LOG.warn(f"Scale factor (right) not found for {self.pipeline.options.subjid} - will output unscaled data")
            sf_right = 1

        t1_dir = "t1_molli_stitch_fix"
        t1_glob = "t1_*.nii.gz"
        t1s = self.inimgs(t1_dir, t1_glob, src=self.OUTPUT)
        if not t1s:
            self.no_data(f"No T1 maps found in {t1_dir} matching {t1_glob}")

        for t1 in t1s:
            left_data = t1.data * sf_left
            right_data = t1.data * sf_right
            t1.save_derived(left_data, self.outfile(t1.fname.replace(".nii.gz", "_scaled_left.nii.gz")))
            t1.save_derived(right_data, self.outfile(t1.fname.replace(".nii.gz", "_scaled_right.nii.gz")))

MODULES = [
    # Parameter maps
    maps.T1Molli(name="t1_molli", molli_dir="../fsort/t1_molli", molli_glob="t1_molli_raw*.nii.gz", t1_thresh=(0, 5000), tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0], tis_use_md=True),
    maps.T1Molli(name="t1_molli_mdr", molli_dir="../fsort/t1_molli_raw", molli_glob="t1_molli_raw*.nii.gz", mdr=True, use_scanner_maps=False, tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0], tis_use_md=True),
    maps.T1Molli(name="t1_molli_nomdr", molli_dir="../fsort/t1_molli_raw", molli_glob="t1_molli_raw*.nii.gz", mdr=False, use_scanner_maps=False, tis=[117.0, 201.0, 1117.0, 1201.0, 2117.0, 2201.0, 3117.0, 4117.0], tis_use_md=True),
    maps.T1SE(name="t1_se_nomdr", se_dir="../fsort/t1_se_raw", tis=np.arange(100, 2001, 100), tss=53.7, mag_only=True),
    maps.T1SE(name="t1_se_mdr", se_dir="t1_se_raw", tis=np.arange(100, 2001, 100), tss=53.7, mdr=True, mag_only=True, parameters=2),
    maps.T1SE(name="t1_se_mdr_step2", se_dir="t1_se_mdr", tis=np.arange(100, 2001, 100), tss=53.7, se_mag_glob="*_reg.nii.gz", mdr=True, mag_only=True, parameters=3, se_src=Module.OUTPUT),
    maps.T2(),
    maps.T2star(),
    maps.MTR(),
    maps.B0(),
    maps.B1(),
    maps.FatFractionDixon(name="ff_dixon_cor", dixon_dir="dixon_cor"),
    maps.FatFractionDixon(name="ff_dixon_ax", dixon_dir="dixon_ax"),
    maps.DwiMoco(),
    maps.DwiAdc(),
    maps.AslMoco(name="pcasl_moco", asl_glob="pcasl*.nii.gz"),
    maps.AslMoco(name="fair_moco", asl_glob="fair*.nii.gz"),

    # Stitch together potentially multiple slice maps
    regrid.StitchSlices(
        name="t1_molli_stitch",
        img_dir="t1_molli",
        imgs={
            "*t1_map*.nii.g?" : "t1_map.nii.gz",  # Just to make sure globs are unique keys!
            "*t1_map*.nii.gz" : "t1_conf.nii.gz",
            "*t1_conf*.nii.gz" : "t1_conf.nii.gz",
        }
    ),
    regrid.StitchSlices(
        name="t1_molli_nomdr_stitch",
        img_dir="t1_molli_nomdr",
        imgs={
            "*t1_map*.nii.gz" : "t1_map.nii.gz",
        }
    ),
    regrid.StitchSlices(
        name="t1_molli_mdr_stitch",
        img_dir="t1_molli_mdr",
        imgs={
            "*t1_map*.nii.gz" : "t1_map.nii.gz",
        }
    ),
    regrid.StitchSlices(
        name="t1_se_nomdr_stitch",
        img_dir="t1_se_nomdr",
        imgs={
            "*t1_map*.nii.gz" : "t1_map.nii.gz",
        }
    ),
    regrid.StitchSlices(
        name="t1_se_mdr_stitch",
        img_dir="t1_se_mdr",
        imgs={
            "*t1_map*.nii.gz" : "t1_map.nii.gz",
        }
    ),
    regrid.StitchSlices(
        name="t1_se_mdr_step2_stitch",
        img_dir="t1_se_mdr_step2",
        imgs={
            "*t1_map*.nii.gz" : "t1_map.nii.gz",
            "*_reg_reg*.nii.gz" : "se_data.nii.gz",
        }
    ),

    # Segmentations
    segmentations.KidneyT1(map_dir="t1_molli_stitch", map_glob="t1_conf.nii.gz", t1_limits=[(0, 0), (4136, 0)]),
    segmentations.KidneyT1(name="seg_kidney_t1_mdr", map_dir="t1_molli_mdr_stitch", map_glob="t1_map.nii.gz", t1_limits=[(0, 0), (4136, 0)]),
    segmentations.KidneyT1(name="seg_kidney_t1_nomdr", map_dir="t1_molli_nomdr_stitch", map_glob="t1_map.nii.gz", t1_limits=[(0, 0), (4136, 0)]),
    segmentations.KidneyT1SE(name="seg_kidney_t1_se", t1_se_dir="t1_se_mdr_step2_stitch", t1_se_glob="*se_data.nii.gz"),
    segmentations.KidneyT2w(),
    segmentations.KidneyCystT2w(t2w_dir="t2w", t2w_glob="t2w.nii.gz", t2w_src=Module.INPUT),
    #segmentations.BodyDixon(),
    segmentations.SatDixon(name="seg_sat_dixon_cor", dixon_dir="dixon_cor"),
    segmentations.LiverDixon(name="seg_liver_dixon_cor", dixon_dir="dixon_cor"),
    segmentations.SpleenDixon(name="seg_spleen_dixon_cor",dixon_dir="dixon_cor"),
    segmentations.KidneyDixon(name="seg_kidney_dixon_cor", dixon_dir="dixon_cor", model_id="422"),
    segmentations.SatDixon(name="seg_sat_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.LiverDixon(name="seg_liver_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.SpleenDixon(name="seg_spleen_dixon_ax", dixon_dir="dixon_ax"),
    segmentations.KidneyDixon(name="seg_kidney_dixon_ax", dixon_dir="dixon_ax", model_id="422"),
    segmentations.PancreasEthrive(),
    segmentations.BodyDixon(name="seg_body_dixon_ax", dixon_dir="dixon_ax"),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive", "pancreas.nii.gz"),
    segmentations.VatDixon(
        name="seg_vat_dixon_ax",
        ff_dir="ff_dixon_ax",
        ff_glob="fat_fraction_scanner.nii.gz",
        body_dir="seg_body_dixon_ax",
        sat_dir="seg_sat_dixon_ax",
        organs={
            "seg_liver_dixon_ax" : "liver.nii.gz",
            "seg_spleen_dixon_ax" : "spleen.nii.gz",
            "seg_pancreas_ethrive_largestblob" : "pancreas.nii.gz",
            "seg_kidney_dixon_ax" : "kidney.nii.gz"
        }
    ),

    # Manual fixes
    seg_postprocess.SplitLR(
        "seg_kidney_t1_se",
        "*kidney*.nii.gz",
    ),
    maps.MapFix(
        "t1_molli_stitch",
        fix_dir_option="seg_kidney_t1_fix",
        maps={
            "t1_map.nii.gz" : {
                "glob" : "%s/t1_map.nii.gz",
            },
            "t1_conf.nii.gz" : {
                "glob" : "%s/t1_map.nii.gz",
            },
        },
    ),
    T1Scaled(),
    seg_postprocess.SegFix(
        "seg_kidney_t1",
        fix_dir_option="seg_kidney_t1_fix",
        segs={
            "*cortex_l*.nii.gz" : {
                "glob" : "%s/*cortex*.nii.gz",
                "side" : "left",
                "fname" : "kidney_cortex_l.nii.gz",
            },
            "*cortex_r*.nii.gz" : {
                "glob" : "%s/*cortex*.nii.gz",
                "side" : "right",
                "fname" : "kidney_cortex_r.nii.gz",
            },
            "*medulla_l*.nii.gz" : {
                "glob" : "%s/*medulla*.nii.gz",
                "side" : "left",
                "fname" : "kidney_medulla_l.nii.gz",
            },
            "*medulla_r*.nii.gz" : {
                "glob" : "%s/*medulla*.nii.gz",
                "side" : "right",
                "fname" : "kidney_medulla_r.nii.gz",
            },
        },
        map_dir="t1_molli_stitch_fix",
        map_fname="t1_map.nii.gz"
    ),
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
            "seg_kidney_t1_fix" : "kidney*.nii.gz",
            "t1_molli_stitch_fix" : "t1_conf.nii.gz",
        }
    ),

    # Segmentation cleaning
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean",
        srcdir="seg_kidney_t1_align_t2star",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="seg_kidney_t1_align_t2star",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean_native",
        srcdir="seg_kidney_t1_fix",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_stitch_fix",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_mdr_clean_native",
        srcdir="seg_kidney_t1_mdr",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_mdr_stitch",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_se_clean_native",
        srcdir="seg_kidney_t1_se_splitlr",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_se_mdr_step2_stitch",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w_fix",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_clean_native_generic",
        srcdir="seg_kidney_t1_fix",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_molli_stitch_fix",
        t1_map_glob="t1_map.nii.gz",
        t2w=False
    ),
    seg_postprocess.KidneyCystClean(
        cyst_dir="seg_kidney_cyst_t2w_fix",
        seg_t2w_dir="seg_kidney_t2w_fix",
        t2w_dir="t2w",
        t2w_glob="t2w.nii.gz",
        t2w_src=Module.INPUT,
    ),
    seg_postprocess.SegVolumes(
        "seg_kidney_t2w_vols",
        seg_dir="seg_kidney_t2w_fix",
        segs={
            "kv_left" : "*left*.nii.gz",
            "kv_right" : "*right*.nii.gz",
            "kv_mask" : "*mask*.nii.gz",
        }
    ),
    segmentations.KidneyCortexMedullaT2w(
        t2w_seg_dir="seg_kidney_t2w_fix",
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_ax",
        ff_dir="ff_dixon_ax",
        ff_glob="fat_fraction_scanner.nii.gz",
        kidney_seg_dir="seg_kidney_dixon_ax",
        kidney_seg_glob="kidney.nii.gz",
        ff_thresh=15,
    ),
    segmentations.KidneyFat(
        name="seg_kidney_fat_dixon_cor",
        ff_dir="ff_dixon_cor",
        ff_glob="fat_fraction_scanner.nii.gz",
        kidney_seg_dir="seg_kidney_dixon_cor",
        kidney_seg_glob="kidney.nii.gz",
        ff_thresh=15,
    ),
    segmentations.RenalPelvis(
        t2w_seg_dir="seg_kidney_t2w_fix",
        t1_seg_dir="seg_kidney_t1_clean_native",
    ),

    # Statistics and numerical measures
    Stats(),
    StatsDixon(),
    statistics.CMD(
        cmd_params=["t2star", "t2star", "t1", "mtr"],
        skip_params=["t1_noclean"],
    ),
    T1MolliMetadata(),
    statistics.ShapeMetrics(
        name="tkv_shape_metrics",
        seg_dir="seg_kidney_t2w_fix",
        segs={
            "tkv_l" : "*left*.nii.gz",
            "tkv_r" : "*right*.nii.gz"
        },
        metrics = ["surf_area", "surf_area_over_vol", "vol", "compactness", "long_axis", "short_axis", "mi1", "mi2", "mi3", "mi_mean", "fa"]
    ),
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_cor",
        seg_dir="seg_kidney_dixon_cor",
        segs={
            "kidney_left_cor" : "kidney_left.nii.gz",
            "kidney_right_cor" : "kidney_right.nii.gz"
        },
        metrics = ["surf_area", "surf_area_over_vol", "vol", "compactness", "long_axis", "short_axis", "mi1", "mi2", "mi3", "mi_mean", "fa"]
    ),
    statistics.ShapeMetrics(
        name="kidney_dixon_shape_metrics_ax",
        seg_dir="seg_kidney_dixon_ax",
        segs={
            "kidney_left_ax" : "kidney_left.nii.gz",
            "kidney_right_ax" : "kidney_right.nii.gz"
        },
        metrics = ["surf_area", "surf_area_over_vol", "vol", "compactness", "long_axis", "short_axis", "mi1", "mi2", "mi3", "mi_mean", "fa"]
    ),
    statistics.Radiomics(
        name="tkv_radiomics",
        params={
            "t2w" : {"dir" : "t2w", "fname" : "t2w.nii.gz", "src" : Module.INPUT},
        },
        segs={
            "tkv_l" : {"dir" : "seg_kidney_t2w_fix", "fname" : "*left*.nii.gz"},
            "tkv_r" : {"dir" : "seg_kidney_t2w_fix", "fname" : "*right*.nii.gz"},
        },
        features={
            "shape" : ["SurfaceArea", "VoxelVolume", "SurfaceVolumeRatio", "MajorAxisLength", "MinorAxisLength", "Elongation", "Compactness1"],
        },
    ),
    statistics.Radiomics(
        name="kidney_dixon_radiomics_cor",
        params={
            "t1" : {"dir" : "t1_molli_stitch_fix", "fname" : "t1_map.nii.gz", "src" : Module.OUTPUT},
        },
        segs={
            "kidney_dixon_l_cor" : {"dir" : "seg_kidney_dixon_cor", "fname" : "*left*.nii.gz"},
            "kidney_dixon_r_cor" : {"dir" : "seg_kidney_dixon_cor", "fname" : "*right*.nii.gz"},
        },
        features={
            "shape" : ["SurfaceArea", "VoxelVolume", "SurfaceVolumeRatio", "MajorAxisLength", "MinorAxisLength", "Elongation", "Compactness1"],
        },
    ),
    statistics.Radiomics(
        name="kidney_dixon_radiomics_ax",
        params={
            "t1" : {"dir" : "t1_molli_stitch_fix", "fname" : "t1_map.nii.gz", "src" : Module.OUTPUT},
        },
        segs={
            "kidney_dixon_l_ax" : {"dir" : "seg_kidney_dixon_ax", "fname" : "*left*.nii.gz"},
            "kidney_dixon_r_ax" : {"dir" : "seg_kidney_dixon_ax", "fname" : "*right*.nii.gz"},
        },

        features={
            "shape" : ["SurfaceArea", "VoxelVolume", "SurfaceVolumeRatio", "MajorAxisLength", "MinorAxisLength", "Elongation", "Compactness1"],
        },
    ),
    statistics.Radiomics(
        name="t1_radiomics_left",
        params={
            "t1" : {"dir" : "t1_scaled", "fname" : "t1_conf_scaled_left.nii.gz"},
            "t2_exp" : {"dir" : "t2", "fname" : "t2_exp.nii.gz"},
            "t2_stim" : {"dir" : "t2", "fname" : "t2_stim.nii.gz"},
        },
        segs={
            "cortex_l" : {"dir" : "seg_kidney_t1_clean_native", "fname" : "*cortex_l*.nii.gz"},
            "medulla_l" : {"dir" : "seg_kidney_t1_clean_native", "fname" : "*medulla_l*.nii.gz"},
        },
    ),
    statistics.Radiomics(
        name="t1_radiomics_right",
        params={
            "t1" : {"dir" : "t1_scaled", "fname" : "t1_conf_scaled_right.nii.gz"},
            "t2_exp" : {"dir" : "t2", "fname" : "t2_exp.nii.gz"},
            "t2_stim" : {"dir" : "t2", "fname" : "t2_stim.nii.gz"},
        },
        segs={
            "cortex_r" : {"dir" : "seg_kidney_t1_clean_native", "fname" : "*cortex_r*.nii.gz"},
            "medulla_r" : {"dir" : "seg_kidney_t1_clean_native", "fname" : "*medulla_r*.nii.gz"},
        },
    ),
    statistics.ISNR(
        src=Module.INPUT,
        imgs={
            "t1w" : "t1w.nii.gz",
            "t2w" : "t2w.nii.gz",
        }
    ),
]

def add_options(parser):
    parser.add_argument("--t1-scale-factors", help="Directory containing manual fixed cyst masks")
    parser.add_argument("--cyst-masks", help="Directory containing manual fixed cyst masks")
    parser.add_argument("--seg-kidney-t2w-fix", help="Directory containing manual T2w kidney masks")
    parser.add_argument("--seg-kidney-t1-fix", help="Directory containing manual kidney cortex/medulla masks + maps")
