import collections
import logging

import numpy as np
import scipy

from fsort.image_file import ImageFile
from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, StatsModule
import fproc.stats as stats

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class T1Seg(Module):
    def __init__(self):
        Module.__init__(self, "t1_seg")

    def process(self):
        for map_type in ("mdr", "nomdr"):
            t1map = self.inimg("t1", f"t1_map_{map_type}.nii.gz")
            LOG.info(f" - Segmenting T1 map from {t1map.fname} using model {self.pipeline.options.t1_model}")
            self.runcmd([
                'kidney_t1_seg',
                '--input', t1map.dirname,
                '--subjid', '',
                '--display-id', self.pipeline.options.subjid,
                '--t1', t1map.fname,
                '--model', self.pipeline.options.t1_model,
                '--noclean',
                '--nooverlay',
                '--output', self.outdir,
                '--outprefix', f'seg_kidney_{map_type}'],
                logfile=f'seg_kidney_{map_type}.log'
            )

            # Generate overlay images using T1 map
            seg_img = ImageFile(self.outfile(f"seg_kidney_{map_type}_cortex_t1.nii.gz"))
            self.lightbox(t1map.data, seg_img, f"seg_kidney_{map_type}_cortex_lightbox")
            seg_img = ImageFile(self.outfile(f"seg_kidney_{map_type}_medulla_t1.nii.gz"))
            self.lightbox(t1map.data, seg_img, f"seg_kidney_{map_type}_medulla_lightbox")

class T1Clean(Module):
    def __init__(self):
        Module.__init__(self, "t1_clean")

    def _clean_t2w(self, t1_seg, t2w_mask):
        mask_data = self.resample(t2w_mask, t1_seg, is_roi=True, allow_rotated=True).get_fdata()
        if np.count_nonzero(mask_data) == 0:
            LOG.warn(f" - T2w mask zero voxel count when regridded onto T1. T2w and T1 scans probably do not align")

        # Dilate T2w masks by 2 voxels
        mask_data = scipy.ndimage.binary_dilation(mask_data, structure=np.ones((3, 3, 3)))
        mask_data = scipy.ndimage.binary_dilation(mask_data, structure=np.ones((3, 3, 3)))
        cleaned_data = (t1_seg.data * mask_data).astype(np.uint8)
        LOG.debug(f" - Voxel counts: orig {np.count_nonzero(t1_seg.data)}, mask {np.count_nonzero(t2w_mask.data)}, dil mask {np.count_nonzero(mask_data)}, out {np.count_nonzero(cleaned_data)},")
        return cleaned_data

    def process(self):
        t1_segs = self.inimgs("t1_seg", "seg_kidney_*.nii.gz", is_depfile=True)
        if not t1_segs:
            self.no_data(" - No T1 segmentations found to clean")

        t1_map_mdr = self.inimg("t1", "t1_map_mdr.nii.gz")
        t1_map_nomdr = self.inimg("t1", "t1_map_nomdr.nii.gz")
        t2w_mask = self.inimg("tkv", "tkv_mask.nii.gz")

        for t1_seg in t1_segs:
            LOG.info(f" - Cleaning T1 segmentation {t1_seg.fname} using T2w mask {t2w_mask.fname}")
            cleaned_data_t1_seg = self._clean_t2w(t1_seg, t2w_mask)

            cleaned_basename = t1_seg.fname_noext + "_cleaned"
            t1_seg.save_derived(cleaned_data_t1_seg, self.outfile(cleaned_basename + ".nii.gz"))

            # Generate overlay images using T1 map
            if "nomdr" in t1_seg.fname:
                self.lightbox(t1_map_nomdr.data, cleaned_data_t1_seg, cleaned_basename + "_lightbox")
            else:
                self.lightbox(t1_map_mdr.data, cleaned_data_t1_seg, cleaned_basename + "_lightbox")

class Deformation(Module):
    def __init__(self):
        Module.__init__(self, "def")

    STATS = {
        "n" : stats.n, "iqn" : stats.iqn, "mean" : stats.mean, "iqmean" : stats.iqmean, 
        "std" : stats.std, "iqstd" : stats.iqstd
    }

    def process(self):
        # For max/min Jacobians subtract 1 (so range becomes -1 to +1)
        for fname in ("jac_max", "jac_min", "jac"):
            img = self.inimg("def", f"{fname}.nii.gz")
            data_sub_1 = img.data - 1
            img.save_derived(data_sub_1, self.outfile(f"{fname}_sub.nii.gz"))
        
        def_max = self.inimg("def", "def_max.nii.gz")
        def_norm = self.inimg("def", "def_norm.nii.gz")
        jac_max = self.inimg("def", "jac_max_sub.nii.gz", is_depfile=True)
        jac_min = self.inimg("def", "jac_min_sub.nii.gz", is_depfile=True)
        jac = self.inimg("def", "jac_sub.nii.gz", is_depfile=True)

        LOG.info(f" - Resampling TKV segmentations onto deformation grid from {def_max.fname}")
        masks = {"all" : np.ones(def_max.shape, dtype=int)}
        for mask_name in ("left", "right"):
            masks[mask_name] = self.resample(self.inimg("tkv", f"tkv_{mask_name}.nii.gz"), def_max, is_roi=True, allow_rotated=True).get_fdata()
            LOG.info(f" - {mask_name} has {np.count_nonzero(masks[mask_name])} voxels")

        stats_data = collections.OrderedDict()
        n_slices = def_max.shape[2]
        n_tis = def_norm.shape[3]
        for mask_name, mask in masks.items():
            LOG.info(f" - Processing mask: {mask_name}")
            max_def_slice, max_def_slice_idx = 0, -1
            for slice_idx in range(n_slices):
                # For each slice, compute in the TKV kidney mask L and R separately, and for the whole image: 
                #   IQmean and stdev of the Max deformation image.
                #   Iqmean and stdev: 
                #     Max Jacobian image (positive voxels and negative voxels and the number of each)
                #     Min Jacobian image (positive voxels and negative voxels and the number of each)
                LOG.info(f"   - slice: {slice_idx}")
                def_max_slicedata = def_max.data[..., slice_idx][mask[..., slice_idx] > 0]
                jac_max_slicedata = jac_max.data[..., slice_idx][mask[..., slice_idx] > 0]
                jac_min_slicedata = jac_min.data[..., slice_idx][mask[..., slice_idx] > 0]
                jac_max_slicedata_pos = jac_max_slicedata[jac_max_slicedata >= 0]
                jac_max_slicedata_neg = jac_max_slicedata[jac_max_slicedata < 0]
                jac_min_slicedata_pos = jac_min_slicedata[jac_min_slicedata >= 0]
                jac_min_slicedata_neg = jac_min_slicedata[jac_min_slicedata < 0]
                max_def = stats.mean(def_max_slicedata)
                if max_def_slice_idx == -1 or max_def_slice < max_def:
                    max_def_slice, max_def_slice_idx = max_def, slice_idx
                for data_name, arr in {
                    "def_max" : def_max_slicedata, "jac_max_pos" : jac_max_slicedata_pos,
                    "jac_max_neg" : jac_max_slicedata_neg, "jac_min_pos" : jac_min_slicedata_pos,
                    "jac_min_neg" : jac_min_slicedata_neg,
                }.items():
                    for stat_name, impl in self.STATS.items():
                        stats_data[f"{data_name}_s{slice_idx}_{mask_name}_{stat_name}"] = impl(arr)

            # For the slice with greatest Max deformation: 
            # Then use the deformation norm file and Jacobian file, estimate for each TI:
            LOG.info(f" - Max deformation found in slice: {max_def_slice_idx}: {max_def_slice}")
            stats_data[f"def_max_slice_idx_{mask_name}"] = max_def_slice_idx
            for ti_idx in range(n_tis):
                # Max deformation, Max Jacobian and Min Jacobian.
                # Max Jacobian image (positive voxels and negative voxels and the number of each)
                # Min Jacobian image (positive voxels and negative voxels and the number of each)
                LOG.info(f"   - TI: {ti_idx}")
                def_norm_tidata = def_norm.data[..., max_def_slice_idx, ti_idx][mask[..., max_def_slice_idx] > 0]
                jac_tidata = jac.data[..., max_def_slice_idx, ti_idx][mask[..., max_def_slice_idx] > 0]
                jac_tidata_pos = jac_tidata[jac_tidata >= 0]
                jac_tidata_neg = jac_tidata[jac_tidata < 0]
                for data_name, arr in {
                    "def_norm" : def_norm_tidata, "jac_pos" : jac_tidata_pos, "jac_neg" : jac_tidata_neg
                }.items():
                    for stat_name, impl in self.STATS.items():
                        stats_data[f"{data_name}_t{ti_idx}_{mask_name}_{stat_name}"] = impl(arr)

        stats_path = self.outfile("stats.csv")
        LOG.info(f" - Saving stats to {stats_path}")
        with open(stats_path, "w") as stats_file:
            for name, value in stats_data.items():
                stats_file.write(f"{name},{str(value)}\n")

class Stats(StatsModule):
    def __init__(self):
        StatsModule.__init__(self,
            segs={
                "cortex_l_nomdr" : {
                    "dir" : "t1_seg",
                    "glob" : "seg_kidney_nomdr_cortex_l_t1.nii.gz"
                },
                "medulla_l_nomdr" : {
                    "dir" : "t1_seg",
                    "glob" : "seg_kidney_nomdr_medulla_l_t1.nii.gz"
                },
                "cortex_l_mdr" : {
                    "dir" : "t1_seg",
                    "glob" : "seg_kidney_mdr_cortex_l_t1.nii.gz"
                },
                "medulla_l_mdr" : {
                    "dir" : "t1_seg",
                    "glob" : "seg_kidney_mdr_medulla_l_t1.nii.gz"
                },
                "cortex_r_nomdr" : {
                    "dir" : "t1_seg",
                    "glob" : "seg_kidney_nomdr_cortex_r_t1.nii.gz"
                },
                "medulla_r_nomdr" : {
                    "dir" : "t1_seg",
                    "glob" : "seg_kidney_nomdr_medulla_r_t1.nii.gz"
                },
                "cortex_r_mdr" : {
                    "dir" : "t1_seg",
                    "glob" : "seg_kidney_mdr_cortex_r_t1.nii.gz"
                },
                "medulla_r_mdr" : {
                    "dir" : "t1_seg",
                    "glob" : "seg_kidney_mdr_medulla_r_t1.nii.gz"
                },
                "cortex_l_nomdr_cleaned" : {
                    "dir" : "t1_clean",
                    "glob" : "seg_kidney_nomdr_cortex_l_t1_cleaned.nii.gz"
                },
                "medulla_l_nomdr_cleaned" : {
                    "dir" : "t1_clean",
                    "glob" : "seg_kidney_nomdr_medulla_l_t1_cleaned.nii.gz"
                },
                "cortex_l_mdr_cleaned" : {
                    "dir" : "t1_clean",
                    "glob" : "seg_kidney_mdr_cortex_l_t1_cleaned.nii.gz"
                },
                "medulla_l_mdr_cleaned" : {
                    "dir" : "t1_clean",
                    "glob" : "seg_kidney_mdr_medulla_l_t1_cleaned.nii.gz"
                },
                "cortex_r_nomdr_cleaned" : {
                    "dir" : "t1_clean",
                    "glob" : "seg_kidney_nomdr_cortex_r_t1_cleaned.nii.gz"
                },
                "medulla_r_nomdr_cleaned" : {
                    "dir" : "t1_clean",
                    "glob" : "seg_kidney_nomdr_medulla_r_t1_cleaned.nii.gz"
                },
                "cortex_r_mdr_cleaned" : {
                    "dir" : "t1_clean",
                    "glob" : "seg_kidney_mdr_cortex_r_t1_cleaned.nii.gz"
                },
                "medulla_r_mdr_cleaned" : {
                    "dir" : "t1_clean",
                    "glob" : "seg_kidney_mdr_medulla_r_t1_cleaned.nii.gz"
                },
            }, 
            params={
                "t1_mdr" : {
                    "dir" : "t1",
                    "src" : "INPUT",
                    "glob" : "t1_map_mdr.nii.gz"
                },
                "t1_nomdr" : {
                    "dir" : "t1",
                    "src" : "INPUT",
                    "glob" : "t1_map_nomdr.nii.gz"
                },
            }, 
            stats=["iqmean", "median", "iqstd"],
            seg_volumes=True,
        )

MODULES = [
    T1Seg(),
    T1Clean(),
    Deformation(),
    Stats(),
]

class MdrCompareArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "mdr_compare", __version__)
        self.add_argument("--t1-model", help="Filename or URL for T1 segmentation model weights", default="/software/imaging/ukbbseg/ukbb-mri-sseg/trained_models/kidney_t1_min_max.pt")

class MdrCompare(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "mdr_compare", __version__, MdrCompareArgumentParser(), MODULES)

if __name__ == "__main__":
    MdrCompare().run()
