import glob
import logging
import os

import numpy as np
import scipy.ndimage

from fsort.image_file import ImageFile
from fproc.module import Module
from fproc.modules import segmentations, seg_postprocess, statistics, maps, regrid

LOG = logging.getLogger(__name__)

class PancreasSegRestricted(Module):
    def __init__(self):
        Module.__init__(self, "seg_pancreas_ethrive_restricted")

    def process(self):
        seg_orig = self.inimg("seg_pancreas_ethrive_fix_largestblob", "pancreas.nii.gz", is_depfile=True)
        ff = self.inimg("fat_fraction", "fat_fraction_scanner.nii.gz", is_depfile=True)
        ff_resamp = self.resample(ff, seg_orig, is_roi=False).get_fdata().squeeze()
        ff_30 = ff_resamp < 30
        ff_50 = ff_resamp < 50
        seg_30 = np.logical_and(seg_orig.data > 0, ff_30)
        seg_50 = np.logical_and(seg_orig.data > 0, ff_50)
        LOG.info(" - Saving pancreas masks restricted by fat fraction")
        seg_orig.save_derived(ff_resamp, self.outfile("fat_fraction.nii.gz"))
        seg_orig.save_derived(ff_30, self.outfile("fat_fraction_lt_30.nii.gz"))
        seg_orig.save_derived(ff_50, self.outfile("fat_fraction_lt_50.nii.gz"))
        seg_orig.save_derived(seg_30, self.outfile("seg_pancreas_ff_lt_30.nii.gz"))
        seg_orig.save_derived(seg_50, self.outfile("seg_pancreas_ff_lt_50.nii.gz"))

class T1Molli(Module):
    def __init__(self):
        Module.__init__(self, "t1_molli")

    def process(self):
        add_niftis = self.pipeline.options.add_niftis
        base_subjid = self.pipeline.options.subjid
        t1s = os.path.join(add_niftis, base_subjid + "*")
        t1 = self.single_inimg("molli_t1_map_nifti", "*.nii.gz", src=t1s)
        if t1:
            LOG.info(f" - Saving MOLLI T1 map from {t1.fname}")
            map = t1.data[..., 0]
            conf = t1.data[..., 1]
            t1.save_derived(map, self.outfile("t1_map.nii.gz"))
            t1.save_derived(map, self.outfile("t1_conf.nii.gz"))

class T1SE(Module):
    def __init__(self):
        Module.__init__(self, "t1_se")

    def process(self):
        add_niftis = self.pipeline.options.add_niftis
        base_subjid = self.pipeline.options.subjid
        t1s = os.path.join(add_niftis, base_subjid + "*")
        t1 = self.single_inimg("seepi_t1_map_nifti", "*.nii.gz", src=t1s)
        if t1:
            LOG.info(f" - Saving SE T1 map from {t1.fname}")
            t1.save(self.outfile("t1.nii.gz"))

class Radiomics(statistics.Radiomics):
    def __init__(self):
        statistics.Radiomics.__init__(
            self,
            params={
                "t1_molli" : {"dir" : "t1_molli", "fname" : "t1_conf.nii.gz", "minval" : 200, "maxval" : 1400},
                "t1_se" : {"dir" : "t1_se", "fname" : "t1.nii.gz", "minval" : 200, "maxval" : 1400},
            },
            segs = {
                "liver" : {"dir" : "seg_liver_dixon_fix", "fname" : "liver.nii.gz"},
            }
        )

class KidneyStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="kidney_stats",
            default_limits="3t",
            segs = {
                "kidney_cortex" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*cortex*.nii.gz",
                },
                "kidney_cortex_l" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*cortex_l*.nii.gz",
                },
                "kidney_cortex_r" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*cortex_r*.nii.gz",
                },
                "kidney_medulla" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*medulla*.nii.gz",
                },
                "kidney_medulla_l" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*medulla_l*.nii.gz",
                },
                "kidney_medulla_r" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*medulla_r*.nii.gz",
                },
            },
            params = {
                "t1_se" : {
                    "dir" : "t1_se",
                    "glob" : "t1.nii.gz",
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
            },
            stats=["n", "vol", "iqn", "iqvol", "iqmean", "median", "iqstd"],
        )

class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats",
            default_limits="3t",
            segs={
                "liver" : {
                    "dir" : "seg_liver_dixon_fix",
                    "glob" : "liver.nii.gz"
                },
                "spleen" : {
                    "dir" : "seg_spleen_dixon",
                    "glob" : "spleen.nii.gz"
                },
                "kidney_dixon" : {
                    "dir" : "seg_kidney_dixon",
                    "glob" : "kidney.nii.gz"
                },
                "pancreas" : {
                    "dir" : "seg_pancreas_ethrive_fix",
                    "glob" : "pancreas.nii.gz",
                },
                "sat" : {
                    "dir" : "seg_sat_dixon",
                    "glob" : "sat.nii.gz",
                    "params" : [],
                },
                "vat" : {
                    "dir" : "seg_vat_dixon",
                    "glob" : "vat.nii.gz",
                    "params" : [],
                },
                "kidney_t2w" : {
                    "dir" : "seg_kidney_t2w",
                    "glob" : "kidney_mask.nii.gz",
                },
                "kidney_cortex" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*cortex*.nii.gz",
                },
                "kidney_cortex_l" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*cortex_l*.nii.gz",
                },
                "kidney_cortex_r" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*cortex_r*.nii.gz",
                },
                "kidney_medulla" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*medulla*.nii.gz",
                },
                "kidney_medulla_l" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*medulla_l*.nii.gz",
                },
                "kidney_medulla_r" : {
                    "dir" : "seg_kidney_t1_se_clean_native",
                    "glob" : "*medulla_r*.nii.gz",
                },
            },
            params={
                "t2star" : {
                    "dir" : "t2star_dixon",
                    "glob" : "t2star_exclude_fill.nii.gz",
                },
                "ff" : {
                    "dir" : "fat_fraction",
                    "glob" : "fat_fraction_scanner.nii.gz",
                },
                "t1_molli" : {
                    "dir" : "t1_molli",
                    "glob" : "t1_conf.nii.gz",
                },
                "t1_se" : {
                    "dir" : "t1_se",
                    "glob" : "t1.nii.gz",
                },
                "adc" : {
                    "dir" : "adc",
                    "glob" : "adc.nii.gz",
                },
                "mre" : {
                    "dir" : "mre_noconf",
                    "glob" : "mre_noconf.nii.gz",
                    "limits" : (1e-5, None),  # Remove zeros from confidence removal
                },
                "mre_qiba" : {
                    "dir" : "mre_qiba_noconf",
                    "glob" : "mre_qiba_noconf.nii.gz",
                    "limits" : (1e-5, None),  # Remove zeros from confidence removal
                },
                "t2_scanner" : {
                    "dir" : "../t2_map",
                    "glob" : "t2_map.nii.gz",
                },
                "t2_stim" : {
                    "dir" : "t2",
                    "glob" : "t2_stim.nii.gz",
                },
                "b1_stim" : {
                    "dir" : "t2",
                    "glob" : "b1_stim.nii.gz",
                },
                "mtr" : {
                    "dir" : "mtr",
                    "glob" : "mtr.nii.gz",
                },
            },
            stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )


class ADC(Module):
    def __init__(self):
        Module.__init__(self, "adc")

    def process(self):
        adc_img = self.single_inimg("../adc", "adc.nii.gz", src=self.OUTPUT)
        add_adc_dir = self.pipeline.options.add_niftis
        if adc_img:
            LOG.info(f" - Saving ADC map from XNAT: {adc_img.fname}")
            adc_img.save(self.outfile("adc.nii.gz"))
        else:
            LOG.info(f" - No ADC map found in XNAT, looking for additional ADC maps in {add_adc_dir}")
            subjdir = os.path.join(add_adc_dir, self.pipeline.options.subjid, "adc_map")
            adc_fnames = list(glob.glob(os.path.join(subjdir, "*.nii.gz")))
            if adc_fnames:
                if len(adc_fnames) > 1:
                    LOG.warning(f"Found multiple ADC images:  {adc_fnames} - using first")
                adc_fname = adc_fnames[0]
                LOG.info(f" - Saving ADC map from {adc_fname}")
                adc_img = ImageFile(adc_fname, warn_json=False)
                adc_img.save(self.outfile("adc.nii.gz"))

class MRE(Module):
    def __init__(self, name):
        Module.__init__(self, name)

    def process(self):
        img = self.single_inimg(f"../{self.name}", f"{self.name}.nii.gz", src=self.OUTPUT)
        add_dir = self.pipeline.options.add_niftis
        if img is not None:
            LOG.info(f" - Saving {self.name.upper()} map from XNAT: {img.fname}")
            img.save(self.outfile(f"{self.name}.nii.gz"))
        else:
            LOG.info(f" - No {self.name.upper()} map found in XNAT, looking for additional maps in {add_dir}")
            subjdir = os.path.join(add_dir, self.pipeline.options.subjid, f"{self.name}_map")
            fnames = list(glob.glob(os.path.join(subjdir, "*.nii.gz")))
            if fnames:
                if len(fnames) > 1:
                    LOG.warning(f"Found multiple images:  {fnames} - using first")
                fname = fnames[0]
                LOG.info(f" - Saving {self.name.upper()} map from {fname}")
                img = ImageFile(fname, warn_json=False)
                img.save(self.outfile(f"{self.name}.nii.gz"))

class MRERemoveConf(Module):
    def __init__(self, name="mre_noconf", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        mre_dir = self.kwargs.get("mre_dir", "../mre")
        mre_glob = self.kwargs.get("mre_glob", "mre*.nii.gz")
        mre_imgs = self.inimgs(mre_dir, mre_glob, is_depfile=True)
        for img in mre_imgs:
            LOG.info(f" - Removing confidence map from MRE image {img.fname}")
            if img.data.ndim > 3:
                LOG.warning(f" - MRE image {img.fname} has more than 3 dimensions, cannot remove confidence map")
                continue

            data = img.data
            clean = np.zeros(data.shape, dtype=np.float32)
            for vol in range(img.shape[-1]):
                sl = np.array(data[..., vol])
                sl[sl > 0] = 1
                px = scipy.ndimage.prewitt(sl, axis=0)
                py = scipy.ndimage.prewitt(sl, axis=1)
                mag = np.sqrt(px**2 + py**2)
                mag[mag == 0] = 1
                mag[mag > 1] = 0
                clean[..., vol] = mag

            clean = clean * data
            clean[clean < 0] = 0
            img.save_derived(clean, self.outfile(img.fname.replace(".nii.gz", "_noconf.nii.gz")))

__version__ = "0.0.1"

NAME = "resus"

MODULES = [
    maps.DixonClassify(dixon_src="../raw_dixon"),

    # Segmentations
    segmentations.BodyDixon(),
    segmentations.SatDixon(),
    segmentations.LiverDixon(),
    segmentations.SpleenDixon(),
    segmentations.KidneyDixon(model_id="422"),
    segmentations.PancreasEthrive(),
    segmentations.KidneyT2w(),
    segmentations.TotalSeg(src_dir="fproc/dixon_classify", dilate=1),

    # Parameter maps
    maps.FatFractionDixon(dixon_dir="fproc/dixon_classify"),
    maps.T2starDixon(),
    ADC(),
    T1Molli(),
    T1SE(),
    maps.MTR(),
    MRE(name="mre"),
    MRE(name="mre_qiba"),
    MRERemoveConf(name="mre_noconf", mre_dir="mre"),
    MRERemoveConf(name="mre_qiba_noconf", mre_dir="mre_qiba"),
    maps.T2(),

    # Post-processing of segmentations
    seg_postprocess.SegFix(
        "seg_pancreas_ethrive",
        fix_dir_option="pancreas_masks",
        segs={
            "pancreas.nii.gz" : {
                "glob" : "%s_*.nii.gz",
                "fname" : "pancreas.nii.gz",
            }
        },
        map_dir="../dixon",
        map_fname="water.nii.gz"
    ),
    seg_postprocess.SegFix(
        "seg_liver_dixon",
        fix_dir_option="liver_masks",
        segs={
            "liver.nii.gz" : {
                "glob" : "%s_*.nii.gz",
                "fname" : "liver.nii.gz",
            }
        },
        map_dir="../dixon",
        map_fname="water.nii.gz"
    ),
    seg_postprocess.LargestBlob("seg_pancreas_ethrive_fix", "pancreas.nii.gz"),
    PancreasSegRestricted(),
    segmentations.VatDixon(
        ff_glob="fat_fraction_scanner.nii.gz",
        fail_on_missing=False,
        organs={
            "seg_liver_dixon_fix" : "liver.nii.gz",
            "seg_spleen_dixon" : "spleen.nii.gz",
            "seg_pancreas_ethrive_fix_largestblob" : "pancreas.nii.gz",
            "seg_kidney_dixon" : "kidney.nii.gz"
        }
    ),

    # This is the T1-SE pipeline from Afirm
    maps.T1SE(name="t1_se_nomdr", se_dir="t1_se_raw", tis=np.arange(100, 2001, 100), tss=53.7, mag_only=True),
    maps.T1SE(name="t1_se_mdr", se_dir="t1_se_raw", tis=np.arange(100, 2001, 100), tss=53.7, mdr=True, mag_only=True, parameters=2),
    maps.T1SE(name="t1_se_mdr_step2", se_dir="t1_se_mdr", tis=np.arange(100, 2001, 100), tss=53.7, se_mag_glob="*_reg.nii.gz", mdr=True, mag_only=True, parameters=3, se_src=Module.OUTPUT),
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
    segmentations.KidneyT1SE(
        name="seg_kidney_t1_se",
        t1_se_dir="t1_se_mdr_step2_stitch",
        t1_se_glob="*se_data.nii.gz",
        t1_ref_dir="t1_se_raw",
    ),
    seg_postprocess.SplitLR(
        "seg_kidney_t1_se",
        "*kidney*.nii.gz",
    ),
    seg_postprocess.KidneyT1Clean(
        name="seg_kidney_t1_se_clean_native",
        srcdir="seg_kidney_t1_se_splitlr",
        seg_t1_glob="kidney*.nii.gz",
        t1_map_srcdir="t1_se_mdr_step2_stitch",
        t1_map_glob="t1_map.nii.gz",
        t2w=True,
        seg_t2w_srcdir="seg_kidney_t2w",
    ),

    # Statistics
    Radiomics(),
    SegStats(),
    KidneyStats(),
]

def add_options(parser):
    parser.add_argument("--add-niftis", help="Dir containing additional NIFTI maps")
    parser.add_argument("--pancreas-masks", help="Directory containing manual pancreas masks")
    parser.add_argument("--liver-masks", help="Directory containing manual liver masks")
