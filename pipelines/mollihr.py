from collections import OrderedDict
import logging

import numpy as np
import mdreg

from fsort.image_file import ImageFile
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
                "dir" : "seg_liver_dixon_fix",
                "glob" : "liver.nii.gz"
            },
            "spleen" : {
                "dir" : "seg_spleen_dixon_fix",
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
            "glob" : "fat_fraction_scanner.nii.gz",
            "limits" : (0, 100),
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

class SegStatsMolli5(statistics.SegStats):
    def __init__(self):
        segs = {
            "liver" : {
                "dir" : "seg_liver_dixon_fix",
                "glob" : "liver.nii.gz"
            },
            "spleen" : {
                "dir" : "seg_spleen_dixon_fix",
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
            "glob" : "fat_fraction_scanner.nii.gz",
            "limits" : (0, 100),
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
                    "dir" : "seg_kidney_t1_molli_5_clean",
                    "glob" : f"kidney_molli_raw_{idx}_5_t1_map_cortex_l_t1.nii.gz"
                },
                f"kidney_cortex_r_{idx}" : {
                    "dir" : "seg_kidney_t1_molli_5_clean",
                    "glob" : f"kidney_molli_raw_{idx}_5_t1_map_cortex_r_t1.nii.gz"
                },
                f"kidney_medulla_l_{idx}" : {
                    "dir" : "seg_kidney_t1_molli_5_clean",
                    "glob" : f"kidney_molli_raw_{idx}_5_t1_map_medulla_l_t1.nii.gz"
                },
                f"kidney_medulla_r_{idx}" : {
                    "dir" : "seg_kidney_t1_molli_5_clean",
                    "glob" : f"kidney_molli_raw_{idx}_5_t1_map_medulla_r_t1.nii.gz"
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
            self, name="stats_molli_5", 
            segs=segs,
            params=params,
            stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

class SegStatsMolli7(statistics.SegStats):
    def __init__(self):
        segs = {
            "liver" : {
                "dir" : "seg_liver_dixon_fix",
                "glob" : "liver.nii.gz"
            },
            "spleen" : {
                "dir" : "seg_spleen_dixon_fix",
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
            "glob" : "fat_fraction_scanner.nii.gz",
            "limits" : (0, 100),
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
                    "dir" : "seg_kidney_t1_molli_7_clean",
                    "glob" : f"*_{idx}_t1_map_cortex_l_*.nii.gz"
                },
                f"kidney_cortex_r_{idx}" : {
                    "dir" : "seg_kidney_t1_molli_7_clean",
                    "glob" : f"*_{idx}_t1_map_cortex_r_*.nii.gz"
                },
                f"kidney_medulla_l_{idx}" : {
                    "dir" : "seg_kidney_t1_molli_7_clean",
                    "glob" : f"*_{idx}_t1_map_medulla_l_*.nii.gz"
                },
                f"kidney_medulla_r_{idx}" : {
                    "dir" : "seg_kidney_t1_molli_7_clean",
                    "glob" : f"*_{idx}_t1_map_medulla_r_*.nii.gz"
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
            self, name="stats_molli_7", 
            segs=segs,
            params=params,
            stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

class T1Moco(Module):
    def __init__(self, name="t1_moco", **kwargs):
        Module.__init__(self, name, **kwargs)
    
    def process(self):
        t1_dir = self.kwargs.get("t1_dir", "t1_molli_5")
        t1_glob = self.kwargs.get("t1_glob", "*_%i_*t1_map.nii.gz")
        t1_maps = []
        t1_ref = None
        for n in range(1, 21):
            t1_map = self.single_inimg(t1_dir, t1_glob % n, src=self.OUTPUT)
            if t1_map:
                t1_maps.append(t1_map)
            else:
                break
        
            # Check consistent
            if t1_map.shape != t1_maps[0].shape:
                self.no_data(f"T1 map {t1_map.fname} has different shape than first T1 map {t1_maps[0].fname} - aborting")
            elif not np.allclose(t1_map.affine, t1_maps[0].affine):
                self.no_data(f"T1 map {t1_map.fname} has different affine than first T1 map {t1_maps[0].fname} - aborting")

            # Only t1_thresh has the metadata?
            if self.kwargs.get("use_t1_thresh", False):
                t1_md = ImageFile(t1_map.fpath.replace("t1_map", "t1_thresh"))
            else:
                t1_md = t1_map
            hr = t1_md.heartrate[0]
            fa = t1_md.flipangle
            if hr == 60 and fa == 35:
                if t1_ref is not None:
                    LOG.warn(f" - Found multiple T1 maps with 60 bpm and 35 degree flip angle: {t1_ref.fname} and {t1_map.fname} - using first one")
                else:
                    LOG.info(f" - Found reference T1 map {t1_map.fname} with {hr} bpm and {fa} degree flip angle")
                    t1_ref = t1_map

        if t1_ref is None:
            self.no_data(f"No T1 maps found in {t1_dir} with 60 bpm and 35 degree flip angle")

        t1_ref_data = t1_ref.data.squeeze(-1) if len(t1_ref.data.shape) == 4 else t1_ref.data
        t1_data_3d = [t1_map.data.squeeze(-1) if len(t1_map.data.shape) == 4 else t1_map.data for t1_map in t1_maps]
        t1_data_stacked = np.stack(t1_data_3d, axis=-1)
        t1_ref.save(self.outfile("t1_ref.nii.gz"))
        t1_ref.save_derived(t1_data_stacked, self.outfile("t1_data_stacked.nii.gz"))

        LOG.info(f" - Doing slicewise MoCo on stacked T1 data")
        num_slices = t1_ref.shape[2]
        moco_data = np.zeros_like(t1_data_stacked)
        print(t1_data_stacked.shape)
        def_field = np.zeros(list(t1_data_stacked.shape) + [2,])
        for sl_idx in range(num_slices):
            def _ref_slice(*args, **kwargs):
                slice_data = t1_ref_data[..., sl_idx]
                fit = np.repeat(slice_data[...,np.newaxis], len(t1_maps), axis=-1)
                return fit, np.expand_dims(np.zeros_like(slice_data), axis=-1)

            stacked_slices = t1_data_stacked[..., sl_idx, :]
            print(stacked_slices.shape)
            sl_data_moco, sl_def_field, fit, pars = mdreg.fit(stacked_slices, fit_image={"func" : _ref_slice})
            print(sl_data_moco.shape, sl_def_field.shape, fit.shape)
            sl_def_field = np.transpose(sl_def_field, (0, 1, 3, 2))
            moco_data[..., sl_idx, :] = sl_data_moco
            def_field[..., sl_idx, :, :] = sl_def_field

        t1_ref.save_derived(moco_data, self.outfile("t1_data_moco.nii.gz"))
        # for idx, t1_map in enumerate(t1_maps):
        #     data = moco_data[..., idx]
        #     t1_map.save_derived(data, self.outfile(t1_map.fname.replace(".nii.gz", "_moco.nii.gz")))
        LOG.info(f" - Saved MoCo T1 maps")

        t1_ref.save_derived(fit, self.outfile("fit.nii.gz"))
        u = def_field[..., 0]
        v = def_field[..., 1]
        t1_ref.save_derived(u, self.outfile("def_field_u.nii.gz"))
        t1_ref.save_derived(v, self.outfile("def_field_v.nii.gz"))
        LOG.info(f" - Saved MoCo def field")

        LOG.info(f" - Copying kidney segs from reference into output")
        seg_dir = self.kwargs.get("seg_dir", "seg_kidney_t1_molli_5_clean")
        segs = self.inimgs(seg_dir, f"kidney*_{t1_ref.fname_noext}*.nii.gz", src=self.OUTPUT)
        for seg in segs:
            fname = seg.fname.replace("_" + t1_ref.fname_noext, "")
            LOG.info(f" - {seg.fname} -> {fname}")
            seg.save(self.outfile(fname))


__version__ = "0.0.1"

NAME = "mollihr"

class T1Thresh(Module):
    def __init__(self, t1_dir):
        Module.__init__(self, f"{t1_dir}_thresh")
        self.t1_dir = t1_dir
    
    def process(self):
        t1_maps = self.inimgs(self.t1_dir, "*t1_map.nii.gz", src=self.OUTPUT)
        if not t1_maps:
            self.no_data(f"No T1 maps found in {self.t1_dir}")

        for t1_map in t1_maps:
            data = np.copy(t1_map.data)
            r2 = self.single_inimg(self.t1_dir, t1_map.fname.replace("t1_map", "r2"), src=self.OUTPUT)

            data[r2.data < 0.3] = 0
            data[data >= 4000] = 0
            data[data < 0] = 0
            
            data[:20, ...] = 0
            data[-20:, ...] = 0
            data[:, :20, ...] = 0
            data[:, -20:, ...] = 0

            t1_thresh = ImageFile(t1_map.fpath.replace("t1_map", "t1_thresh"))
            LOG.info(f" - Saving {t1_thresh.fname} to ensure we have JSON")
            t1_thresh.save(self.outfile(t1_map.fname))
            
            LOG.info(f" - Saving thresholded version of {t1_map.fname}")
            t1_map.save_derived(data, self.outfile(t1_map.fname))
            
            
MODULES = [
    # Parameter maps
    maps.B0(),
    maps.FatFractionDixon(),
    CopyModule("b1"),
    maps.T2starDixon(),
    T1(),
    MolliRaw5(),
    maps.T1Molli(name="t1_molli_5", molli_dir="molli_raw_5", src=Module.OUTPUT), # Use this as source for T1 seg
    maps.T1Molli(name="t1_molli_7", molli_dir="../fsort/molli_raw"),
    T1Thresh(t1_dir="t1_molli_5"),
    T1Thresh(t1_dir="t1_molli_7"),
    # Segmentations
    segmentations.LiverDixon(),
    segmentations.SpleenDixon(),
    segmentations.KidneyDixon(),
    segmentations.KidneyT1(map_dir="../fsort/molli"),
    segmentations.KidneyT1(name="seg_kidney_t1_molli_5", map_dir="t1_molli_5_thresh", map_glob="*t1_map.nii.gz"),
    segmentations.KidneyT1(name="seg_kidney_t1_molli_7", map_dir="t1_molli_7_thresh", map_glob="*t1_map.nii.gz"),
    # Fixed segs
    seg_postprocess.SegFix(
        "seg_spleen_dixon",
        fix_dir_option="seg_fix",
        segs={
            "spleen.nii.gz" : {
                "glob" : "%s/Spleen/spleen.nii.gz",
            }
        },
        map_dir="dixon",
        map_fname="water.nii.gz",
        map_src=Module.INPUT,
    ),
    seg_postprocess.SegFix(
        "seg_liver_dixon",
        fix_dir_option="seg_fix",
        segs={
            "liver.nii.gz" : {
                "glob" : "%s/Liver/liver.nii.gz",
            }
        },
        map_dir="dixon",
        map_fname="water.nii.gz",
        map_src=Module.INPUT,
    ),
    # Segmentation cleaning
    seg_postprocess.KidneyT1Clean(t2w=False),
    seg_postprocess.KidneyT1Clean(name="seg_kidney_t1_molli_5_clean", srcdir="seg_kidney_t1_molli_5", t1_map_srcdir="t1_molli_5", t2w=False),
    seg_postprocess.KidneyT1Clean(name="seg_kidney_t1_molli_7_clean", srcdir="seg_kidney_t1_molli_7", t1_map_srcdir="t1_molli_7", t2w=False),
    # Alignment of T1maps
    T1Moco(name="seg_kidney_t1_scanner_moco_vol", t1_dir="t1", t1_glob="t1_map_%i.nii.gz", seg_dir="seg_kidney_t1_clean"),
    T1Moco(name="seg_kidney_t1_molli_5_moco_vol", t1_dir="t1_molli_5_thresh", t1_glob="*_%i_5_t1_map.nii.gz", seg_dir="seg_kidney_t1_molli_5_clean"),
    T1Moco(name="seg_kidney_t1_molli_7_moco_vol", t1_dir="t1_molli_7_thresh", t1_glob="*_%i_t1_map.nii.gz", seg_dir="seg_kidney_t1_molli_7_clean"),
    # Statistics
    SegStats(),
    SegStatsMolli5(),
    SegStatsMolli7(),
]

def add_options(parser):
    parser.add_argument("--seg-fix", help="Directory containing manual fixed masks")
