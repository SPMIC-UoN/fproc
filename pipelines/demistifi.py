import logging
import os

import numpy as np
import skimage
from dbdicom.wrappers.skimage import _volume_features

from fsort.image_file import ImageFile
from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, StatsModule

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class VAT(Module):
    """
    Internal fat from DIXON
    """
    def __init__(self):
        Module.__init__(self, "vat")

    def process(self):
        """
        VAT (internal fat)

        i. Abdominal cavity mask - organ masks (liver, spleen, kidneys, lungs, pancreas) can we use pancreas T1w seg regridded onto Dixon?
        ii. Threshold the whole body fat percent image (include >0.9) and fill small holes 
        iii. Final VAT mask = ixii
        """
        abd_cavity = self.inimg("qpdata", "seg_abdominal_cavity_dixon.nii.gz")
        if not abd_cavity:
            self.no_data("No abdominal cavity image found")

        spleen = self.inimg("qpdata", "seg_spleen_dixon.nii.gz", warn=True, check=False)
        kidney_right = self.inimg("qpdata", "seg_kidney_right_dixon.nii.gz", warn=True, check=False)
        kidney_left = self.inimg("qpdata", "seg_kidney_right_dixon.nii.gz", warn=True, check=False)
        lungs = self.inimg("qpdata", "seg_lungs_dixon.nii.gz", warn=True, check=False)
        liver = self.inimg("qpdata", "seg_liver_dixon.nii.gz", warn=True, check=False)
        pancreas = self.inimg("qpdata", "seg_pancreas_t1w.nii.gz", warn=True, check=False)
        if pancreas is not None:
            pancreas = self.resample(pancreas, abd_cavity, is_roi=False, allow_rotated=True)

        organs = None
        for arr in (spleen, kidney_right, kidney_left, lungs, liver, pancreas):
            if arr is None:
                continue
            elif isinstance(arr, ImageFile):
                arr = arr.data
            else:
                arr = arr.get_fdata()

            if arr is None:
                LOG.warn("Could not remove organ data")
                continue
            arr = (arr > 0).astype(np.int32)
            if organs is None:
                organs = np.copy(arr)
            else:
                organs += arr
        organs = (organs > 0).astype(np.int32)
        no_organs = (abd_cavity.data > 0).astype(np.int32) - organs
        no_organs[no_organs < 0] = 0

        fat = self.inimgs("preproc", "*/analysis/fat.percent.nii.gz")
        if not fat:
            self.no_data("No body fat percent image found")
        elif len(fat) > 1:
            self.bad_data("Multiple body fat percent images found")
        fat = (fat[0].data > 0.55).astype(np.int32)

        vat = (no_organs * fat).astype(np.int32)
        abd_cavity.save_derived(vat, self.outfile("vat_nofill.nii.gz"))

        # Fill small holes (1 voxel) slicewise in Z direction
        for z in range(vat.shape[2]):
            inv_vat_slice = np.logical_not(vat[..., z]).astype(np.int32)
            labelled = skimage.measure.label(inv_vat_slice)
            props = skimage.measure.regionprops(labelled)
            for region in props:
                if np.sum(inv_vat_slice[labelled == region.label]) < 2:
                    vat[..., z][labelled == region.label] = 1

        abd_cavity.save_derived(vat, self.outfile("vat.nii.gz"))

        water = self.inimgs("preproc", "*/nifti/water.nii.gz")
        self.lightbox(water[0], vat, "vat_lightbox", tight=True)

class ASAT(Module):
    def __init__(self):
        Module.__init__(self, "asat")

    def process(self):
        """
        ASAT (Abdominal subcutaneous adipose tissue = external tissue)
        
        i. Invert Body cavity mask
        ii. Threshold the whole body fat percent image (include >0.9) and fill small holes
        ii. Bounding box =  upper and lower bounds of the abdominal mask in the superior-inferior (z) direction.
        iv. Final ASAT mask = 1x2x3
        """
        body_mask = self.inimgs("preproc", "*/analysis/mask.body.nii.gz")
        if not body_mask:
            self.no_data("No body mask image found")
        elif len(body_mask) > 1:
            self.bad_data("Multiple body mask images found")

        body_cavity = self.inimg("qpdata", "seg_body_cavity_dixon.nii.gz")
        asat = (body_mask[0].data > 0).astype(np.int32) - (body_cavity.data > 0).astype(np.int32)
        asat[asat < 0] = 0

        fat = self.inimgs("preproc", "*/analysis/fat.percent.nii.gz")
        if not fat:
            self.no_data("No body fat percent image found")
        elif len(fat) > 1:
            self.bad_data("Multiple body fat percent images found")
        fat = (fat[0].data > 0.7).astype(np.int32)
        asat = asat * fat

        abd_cavity = self.inimg("qpdata", "seg_abdominal_cavity_dixon.nii.gz")
        nonzero_slices_in_abd_cavity = [z for z in range(abd_cavity.shape[2]) if np.count_nonzero(abd_cavity.data[..., z]) > 0]
        bb_bottom, bb_top = min(nonzero_slices_in_abd_cavity), max(nonzero_slices_in_abd_cavity)
        LOG.info(f"Abdominal cavity bounding box in Z direction: {bb_bottom} to {bb_top}")

        asat[..., bb_top+1:] = 0
        asat[..., :bb_bottom] = 0
        abd_cavity.save_derived(asat, self.outfile("asat.nii.gz"))

        water = self.inimgs("preproc", "*/nifti/water.nii.gz")
        self.lightbox(water[0], asat, "asat_lightbox", tight=True)

class MolliFitparams(Module):
    def __init__(self):
        Module.__init__(self, "molli_fitparams")

    def process(self):
        fitparams = self.inimg("qpdata", "molli_fitparams.nii.gz")
        if fitparams.nvols != 4:
            self.bad_data(f"Expected fitparams to be 4 volumes - was {fitparams.nvols}")
        
        for idx, name in enumerate(["rsquare", "a", "b", "t1star"]):
            fitparams.save_derived(fitparams.data[..., idx], self.outfile(f"{name}.nii.gz"))

class MolliHr(Module):
    def __init__(self):
        Module.__init__(self, "molli_hr")

    def process(self):
        LOG.info(f" - Saving HR timings to hr_timings.csv")
        with open(self.outfile("hr_timings.csv"), "w") as f:
            for organ in ("kidney", "liver", "pancreas"):
                molli = self.inimg("fsort/molli", f"molli_{organ}.nii.gz", check=False)
                if molli is None:
                    LOG.warn(f"No MOLLI data found for {organ}")
                    continue
                timings = molli.metadata.get("FrameTimesStart", [])
                mean, max = "", ""
                if not timings:
                    LOG.warn(f"No timings found in metadata for {molli.fname}")
                elif len(timings) != 7:
                    LOG.warn(f"Timings should have length 7, was {len(timings)} for {molli.fname}")
                else:
                    LOG.info(f" - Found HR timings in metadata: {timings}")
                    timings = np.array(timings)
                    dt = timings[1:] - timings[:-1]
                    #  Last two images were across two heart beats
                    dt[4] /= 2
                    dt[5] /= 2
                    mean, max = np.mean(dt), np.max(dt)
                f.write(f"{organ}_mean,{mean}\n")
                f.write(f"{organ}_max,{max}\n")

class ShapeMetrics(Module):
    def __init__(self):
        Module.__init__(self, "shape_metrics")

    def process(self):
        METRICS_MAPPING = {
            'Surface area': "surf_area",
            'Volume': "vol",
            'Bounding box volume': "vol_bb",
            'Convex hull volume': "vol_ch",
            'Volume of holes': "vol_holes",
            'Extent': "extent",
            'Solidity': "solidity",
            'Compactness': "compactness",
            'Long axis length': "long_axis",
            'Short axis length': "short_axis",
            'Equivalent diameter': "equiv_diam",
            'Longest caliper diameter': "longest_diam",
            'Maximum depth': "max_depth",
            'Primary moment of inertia': "mi1",
            'Second moment of inertia': "mi2",
            'Third moment of inertia': "mi3",
            'Mean moment of inertia': "mi_mean",
            'Fractional anisotropy of inertia': "fa",
            'QC - Volume check': "volcheck",
        }

        stats = {}
        for seg in [
            "seg_kidney_left_dixon",
            "seg_kidney_right_dixon",
            "seg_liver_dixon",
            "seg_pancreas_t1w",
            "seg_spleen_dixon",
        ]:
            segimgs = self.inimgs("qpdata", f"{seg}.nii.gz")
            if not segimgs:
                LOG.warn(f" - No segmentation matching {seg} found for shape metrics")
                continue
            elif len(segimgs) > 1:
                LOG.warn(f" - Multiple segmentations matching {seg} found - using first")
            segimg = segimgs[0]
            LOG.info(f" - Calculating shape metrics from {segimg.fname}")
            try:
                vol_metrics = _volume_features(segimg.data, affine=segimg.affine)
            except:
                LOG.exception(f"Error getting volume features for {seg}")
                vol_metrics = {}

            print(vol_metrics)
            for name, metric in METRICS_MAPPING.items():
                col_name = f"{seg}_{metric}"
                value, _units = vol_metrics.get(name, ("", ""))
                stats[col_name] = value

        LOG.info(f" - Saving shape metrics to shape_metrics.csv")
        with open(self.outfile("shape_metrics.csv"), "w") as f:
            for name, value in stats.items():
                f.write(f"{name},{value}\n")

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
    VAT(),
    ASAT(),
    ShapeMetrics(),
    MolliHr()
]

class DemistifiPostprocArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "demistifi_postproc", __version__)
        
class DemistifiPostproc(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "demistifi_postproc", __version__, DemistifiPostprocArgumentParser(), MODULES)

if __name__ == "__main__":
    DemistifiPostproc().run()
