import logging

import numpy as np
import scipy

from fsort.image_file import ImageFile
from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, StatsModule

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

class Overlays(Module):
    def __init__(self):
        Module.__init__(self, "overlays")

    def process(self):
        pass

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
    #Overlays(),
    Stats(),
]

class MdrCompareArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "mdr_compare", __version__)
        self.add_argument("--t1-model", help="Filename or URL for T1 segmentation model weights", default="/software/imaging/ukbbseg/ukbb-mri-sseg/trained_models/kidney_t1.pt")

class MdrCompare(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "mdr_compare", __version__, MdrCompareArgumentParser(), MODULES)

if __name__ == "__main__":
    MdrCompare().run()
