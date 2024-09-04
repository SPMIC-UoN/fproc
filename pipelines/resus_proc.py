import glob
import logging
import os

from fsort import ImageFile
from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, CopyModule, StatsModule

import numpy as np
import skimage

from renal_preproc import B0, B1

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class LiverSeg(Module):
    def __init__(self):
        Module.__init__(self, "liver_seg")

    def process(self):
        """nnUNetv2_predict -i /home/myfolder/ -o /home/myoutputfolder/ -d 50 -f all"""
        self.inimg("dixon", "fat.nii.gz").save(self.outfile("liver_0000"))
        self.inimg("dixon", "t2star.nii.gz").save(self.outfile("liver_0001"))
        self.inimg("dixon", "water.nii.gz").save(self.outfile("liver_0002"))

        LOG.info(f" - Segmenting LIVER using mDIXON data in: {self.outdir}")
        self.runcmd([
                'nnUNetv2_predict',
                '-i', self.outdir,
                '-o', self.outdir,
                '-d', '14',
                '-f', 'all',
                '-c', '3d_fullres',
            ],
            logfile=f'seg_dixon_liver_uunet.log'
        )

        seg = self.inimg("liver_seg", "liver.nii.gz", src=self.OUTPUT)
        water = self.inimg("dixon", "water.nii.gz")
        self.lightbox(water, seg, name="liver_lightbox", tight=True)

class KidneySeg(Module):
    def __init__(self):
        Module.__init__(self, "kidney_seg")

    def process(self):
        t1_map = self.inimg("molli_kidney", "t1_map.nii.gz")
        LOG.info(f" - Segmenting KIDNEY using T1 data: {t1_map.fname}")
        self.runcmd([
            'kidney_t1_seg',
            '--input', t1_map.dirname,
            '--subjid', '',
            '--display-id', self.pipeline.options.subjid,
            '--t1', t1_map.fname,
            '--model', self.pipeline.options.t1_model,
            '--noclean',
            '--output', self.outdir,
            '--outprefix', f'seg_kidney'],
            logfile=f'seg_kidney.log'
        )

class KidneySegClean(Module):
    def __init__(self):
        Module.__init__(self, "kidney_seg_clean")

    def _clean_generic(self, t1_seg, t1_segs, remove_small=True):
        # How close to the edge (as a fraction of total pixels) a blob centroid needs to be
        # before it is discarded
        EDGE_VERT_FRACTION = 0.1
        EDGE_HORIZ_FRACTION = 0.1

        # How close to the horizontal centre (as a fraction of total pixels) a blob centroid needs to be
        # before it is discarded (the kidneys should be either side of the central spine)
        CENTRE_FRACTION = 1.0/12

        # Fraction of pixels to use as criteria for small blob removal. The minimum size of a blob
        # is this fraction of the horizontal dimension squared
        SMALL_FRACTION = 1.0/20

        # Get the whole kidney mask matching the segmentation we are cleaning
        mask_img_cor, mask_img_med = None, None
        for seg in t1_segs:
            if seg.fname_noext.endswith("medulla_t1") and seg.affine_matches(t1_seg):
                mask_img_med = seg.data
            elif seg.fname_noext.endswith("cortex_t1") and seg.affine_matches(t1_seg):
                mask_img_cor = seg.data

        if mask_img_cor is None or mask_img_med is None:
            self.bad_data(f"Could not find cortex and medulla mask images matching {t1_seg.fname}")

        mask_img_cor[mask_img_cor<0] = 0
        mask_img_cor[mask_img_cor>0] = 1
        mask_img_med[mask_img_med<0] = 0
        mask_img_med[mask_img_med>0] = 1
        kid_mask = np.logical_or(mask_img_cor, mask_img_med)
        cleaned_data = np.copy(t1_seg.data)

        for slice_idx in range(kid_mask.shape[2]):
            kid_mask_slice = kid_mask[..., slice_idx]
            labelled = skimage.measure.label(kid_mask_slice)
            props = skimage.measure.regionprops(labelled)

            # Remove any central blobs
            for region in props:
                if (region.centroid[0] < kid_mask.shape[0]*(0.5+CENTRE_FRACTION) and 
                    region.centroid[0] > kid_mask.shape[0]*(0.5-CENTRE_FRACTION)):
                    kid_mask_slice[labelled == region.label] = 0

            # Remove any blobs around the edge
            for region in props:
                if (region.centroid[1] < kid_mask.shape[0]*EDGE_HORIZ_FRACTION or
                    region.centroid[1] > kid_mask.shape[0]*(1-EDGE_HORIZ_FRACTION) or
                    region.centroid[0] < kid_mask.shape[1]*EDGE_VERT_FRACTION or
                    region.centroid[0] > kid_mask.shape[1]*(1-EDGE_VERT_FRACTION)):
                    kid_mask_slice[labelled == region.label] = 0

            # Remove any small blobs from one copy
            if remove_small:
                smallblob_thresh = round((kid_mask_slice.shape[0]*SMALL_FRACTION)**2)
                for region in props:
                    if np.sum(kid_mask_slice[labelled == region.label]) < smallblob_thresh:
                        kid_mask_slice[labelled == region.label] = 0

            cleaned_data[..., slice_idx] *= kid_mask_slice

        return cleaned_data.astype(np.uint8)

    def process(self):
        t1_segs = self.inimgs("kidney_seg", "seg_kidney_*.nii.gz", is_depfile=True)
        t1_map = self.inimg("t1_kidney", "t1_map.nii.gz", is_depfile=True)
        if not t1_segs:
            self.no_data(" - No T1 kidney segmentations found to clean")

        for t1_seg in t1_segs:
            cleaned_basename = t1_seg.fname_noext + "_cleaned"
            cleaned_data_t1_seg = self._clean_generic(t1_seg, t1_segs)
            t1_seg.save_derived(cleaned_data_t1_seg, self.outfile(cleaned_basename + ".nii.gz"))

        # Generate overlay image of whole kidney using T1 map
        cleaned_kidney = self.inimg("kidney_seg_clean", "seg_kidney_all_t1_cleaned.nii.gz", is_depfile=True)
        self.lightbox(t1_map.data, cleaned_kidney.data, "seg_kidney_all_t1_cleaned_lightbox")

class SpleenSeg(Module):
    def __init__(self):
        Module.__init__(self, "spleen_seg")

    def process(self):
        self.inimg("dixon", "fat.nii.gz").save(self.outfile("spleen_0000"))
        self.inimg("dixon", "t2star.nii.gz").save(self.outfile("spleen_0001"))
        self.inimg("dixon", "water.nii.gz").save(self.outfile("spleen_0002"))

        LOG.info(f" - Segmenting SPLEEN using mDIXON data in: {self.outdir}")
        self.runcmd([
                'nnUNetv2_predict',
                '-i', self.outdir,
                '-o', self.outdir,
                '-d', '102',
                '-f', 'all',
                '-c', '3d_fullres',
            ],
            logfile=f'seg_dixon_spleen_uunet.log'
        )

        seg = self.inimg("spleen_seg", "spleen.nii.gz", src=self.OUTPUT)
        water = self.inimg("dixon", "water.nii.gz")
        self.lightbox(water, seg, name="spleen_lightbox", tight=True)

class PancreasSeg(Module):
    def __init__(self):
        Module.__init__(self, "pancreas_seg")

    def process(self):
        ethrive = self.inimg("ethrive", "ethrive.nii.gz")
        ethrive.save(self.outfile("pancreas_0000"))

        LOG.info(f" - Segmenting PANCREAS using eTHRIVE data in: {self.outdir}")
        self.runcmd([
                'nnUNetv2_predict',
                '-i', self.outdir,
                '-o', self.outdir,
                '-d', '234',
                '-f', 'all',
                '-c', '3d_fullres',
            ],
            logfile=f'seg_ethrive_pancreas_uunet.log'
        )

        seg = self.inimg("pancreas_seg", "pancreas.nii.gz", src=self.OUTPUT)
        self.lightbox(ethrive, seg, name="ethrive_lightbox", tight=True)

class SatSeg(Module):
    def __init__(self):
        Module.__init__(self, "sat_seg")

    def process(self):
        self.inimg("dixon", "fat.nii.gz").save(self.outfile("sat_0000"))

        LOG.info(f" - Segmenting SAT using mDIXON data in: {self.outdir}")
        self.runcmd([
                'nnUNetv2_predict',
                '-i', self.outdir,
                '-o', self.outdir,
                '-d', '141',
                '-f', 'all',
                '-c', '3d_fullres',
            ],
            logfile=f'seg_dixon_sat_uunet.log'
        )

        seg = self.inimg("sat_seg", "sat.nii.gz", src=self.OUTPUT)
        water = self.inimg("dixon", "water.nii.gz")
        self.lightbox(water, seg, name="sat_lightbox", tight=True)

class PancreasSegFix(Module):
    def __init__(self):
        Module.__init__(self, "pancreas_seg_fix")

    def process(self):
        seg_origs = self.inimgs("pancreas_seg", "pancreas.nii.gz", is_depfile=True)
        if not seg_origs:
            seg_orig = None
        else:
            if len(seg_origs) > 1:
                LOG.warn(" - Multiple pancreas segmentations found - using first")
            seg_orig = seg_origs[0]

        if not self.pipeline.options.pancreas_masks:
            LOG.info(" - No pancreas masks dir specified")
            seg_new = None
        else:
            globexpr = os.path.join(
                self.pipeline.options.pancreas_masks, 
                "%s_*.nii.gz" % self.pipeline.options.subjid
            )
            pancreas_masks = glob.glob(globexpr)
            if not pancreas_masks:
                LOG.info(f" - No fixed pancreas mask for {self.pipeline.options.subjid} in {globexpr}")
                seg_new = None
            else:
                if len(pancreas_masks) > 1:
                    LOG.warn(f" - Multiple pancreas masks found for {self.pipeline.options.subjid}: {pancreas_masks} - using first")
                seg_new = ImageFile(pancreas_masks[0])

        if seg_new is not None:
            LOG.info(f" - Saving manual PANCREAS seg from {seg_new.fname}")
        elif seg_orig is not None:
            LOG.info(f" - Using original PANCREAS seg from {seg_orig.fname}")
            seg_new = seg_orig
        else:
            LOG.warn(f" - No PANCREAS seg found")

        if seg_new is not None:        
            seg_new.save(self.outfile("seg_pancreas.nii.gz"))

            # Overlay onto dixon
            dixon_water = self.inimgs("dixon", "water.nii.gz")
            if not dixon_water:
                LOG.warn(f" - Could not find Dixon water image for overlay")
            else:
                self.lightbox(dixon_water[0], seg_new, name="pancreas_lightbox", tight=True)

class LiverSegFix(Module):
    def __init__(self):
        Module.__init__(self, "liver_seg_fix")

    def process(self):
        seg_origs = self.inimgs("liver_seg", "liver.nii.gz", is_depfile=True)
        if not seg_origs:
            seg_orig = None
        else:
            if len(seg_origs) > 1:
                LOG.warn(" - Multiple liver segmentations found - using first")
            seg_orig = seg_origs[0]

        if not self.pipeline.options.liver_masks:
            LOG.info(" - No liver masks dir specified")
            seg_new = None
        else:
            globexpr = os.path.join(
                self.pipeline.options.liver_masks, 
                "%s_*.nii.gz" % self.pipeline.options.subjid
            )
            liver_masks = glob.glob(globexpr)
            if not liver_masks:
                LOG.info(f" - No fixed liver mask for {self.pipeline.options.subjid} in {globexpr}")
                seg_new = None
            else:
                if len(liver_masks) > 1:
                    LOG.warn(f" - Multiple liver masks found for {self.pipeline.options.subjid}: {liver_masks} - using first")
                seg_new = ImageFile(liver_masks[0])

        if seg_new is not None:
            LOG.info(f" - Saving manual LIVER seg from {seg_new.fname}")
        elif seg_orig is not None:
            LOG.info(f" - Using original LIVER seg from {seg_orig.fname}")
            seg_new = seg_orig
        else:
            LOG.warn(f" - No LIVER seg found")

        if seg_new is not None:        
            seg_new.save(self.outfile("liver.nii.gz"))

            # Overlay onto dixon
            dixon_water = self.inimgs("dixon", "water.nii.gz")
            if not dixon_water:
                LOG.warn(f" - Could not find Dixon water image for overlay")
            else:
                self.lightbox(dixon_water[0], seg_new, name="liver_lightbox", tight=True)

class SatSegFix(Module):
    def __init__(self):
        Module.__init__(self, "sat_seg_fix")

    def process(self):
        seg_origs = self.inimgs("sat_seg", "sat.nii.gz", is_depfile=True)
        if not seg_origs:
            seg_orig = None
        else:
            if len(seg_origs) > 1:
                LOG.warn(" - Multiple SAT segmentations found - using first")
            seg_orig = seg_origs[0]

        if not self.pipeline.options.sat_masks:
            LOG.info(" - No SAT masks dir specified")
            seg_new = None
        else:
            globexpr = os.path.join(
                self.pipeline.options.sat_masks, 
                "%s_*.nii.gz" % self.pipeline.options.subjid
            )
            sat_masks = glob.glob(globexpr)
            if not sat_masks:
                LOG.info(f" - No fixed SAT mask for {self.pipeline.options.subjid} in {globexpr}")
                seg_new = None
            else:
                if len(sat_masks) > 1:
                    LOG.warn(f" - Multiple SAT masks found for {self.pipeline.options.subjid}: {sat_masks} - using first")
                seg_new = ImageFile(sat_masks[0])

        if seg_new is not None:
            LOG.info(f" - Saving manual SAT seg from {seg_new.fname}")
        elif seg_orig is not None:
            LOG.info(f" - Using original SAT seg from {seg_orig.fname}")
            seg_new = seg_orig
        else:
            LOG.warn(f" - No SAT seg found")

        if seg_new is not None:        
            seg_new.save(self.outfile("sat.nii.gz"))

            # Overlay onto dixon
            dixon_water = self.inimgs("dixon", "water.nii.gz")
            if not dixon_water:
                LOG.warn(f" - Could not find Dixon water image for overlay")
            else:
                self.lightbox(dixon_water[0], seg_new, name="sat_lightbox", tight=True)

class SegFix(Module):
    def __init__(self, srcdir, src_glob, fix_dir_option, fix_glob=None):
        Module.__init__(self, f"{srcdir}_fix")
        self.srcdir = srcdir
        self.src_glob = src_glob
        self.fix_dir_option = fix_dir_option
        if fix_glob:
            self.fix_glob = fix_glob
        else:
            self.fix_glob = "%s_*.nii.gz"

    def process(self):
        origs = self.inimgs(self.srcdir, self.src_glob, is_depfile=True)
        if not origs:
            orig = None
        else:
            if len(origs) > 1:
                LOG.warn(f" - Multiple files found matching {self.src_glob}- using first")
            orig = origs[0]

        fix_dir = getattr(self.pipeline.options, self.fix_dir_option, None)
        if not fix_dir:
            LOG.info(" - No fixed files dir specified")
            new = None
        else:
            globexpr = os.path.join(fix_dir, self.fix_glob % self.pipeline.options.subjid)
            news = glob.glob(globexpr)
            if not news:
                LOG.info(f" - No fixed file for {self.pipeline.options.subjid} in {globexpr}")
                new = None
            else:
                if len(news) > 1:
                    LOG.warn(f" - Multiple fixed files found for {self.pipeline.options.subjid}: {news} - using first")
                new = ImageFile(news[0])

        if new is not None:
            LOG.info(f" - Saving fixed file from {new.fname}")
        elif orig is not None:
            LOG.info(f" - Using original file from {orig.fname}")
            new = orig
        else:
            LOG.warn(f" - No original or fixed file found")

        if new is not None:        
            new.save(self.outfile(self.src_glob))

            # Overlay onto dixon
            dixon_water = self.inimgs("dixon", "water.nii.gz")
            if not dixon_water:
                LOG.warn(f" - Could not find Dixon water image for overlay")
            else:
                self.lightbox(dixon_water[0], new, name="sat_lightbox", tight=True)

class FatFraction(Module):
    def __init__(self):
        Module.__init__(self, "fat_fraction")

    def process(self):
        fat = self.inimg("dixon", "fat.nii.gz")
        water = self.inimg("dixon", "water.nii.gz")

        ff = fat.data.astype(np.float32) / (fat.data + water.data)
        fat.save_derived(ff, self.outfile("fat_fraction.nii.gz"))

class T2Star(Module):
    def __init__(self):
        Module.__init__(self, "t2star")

    def process(self):
        self.copyinput("dixon", "t2star.nii.gz")
 
class T2(CopyModule):
    def __init__(self):
        CopyModule.__init__(self, "t2")

class T1Liver(Module):
    def __init__(self):
        Module.__init__(self, "t1_liver")

    def process(self):
        t1_map = self.inimg("molli_liver", "t1_map.nii.gz")
        t1_map.save(self.outfile("t1_map.nii.gz"))

class T1Kidney(Module):
    def __init__(self):
        Module.__init__(self, "t1_kidney")

    def process(self):
        t1_map = self.inimg("molli_kidney", "t1_map.nii.gz")
        t1_map.save(self.outfile("t1_map.nii.gz"))
        t1_conf = self.inimg("molli_kidney", "t1_conf.nii.gz")
        t1_conf.save(self.outfile("t1_conf.nii.gz"))

class SeT1Map(Module):
    def __init__(self):
        Module.__init__(self, "se_t1")

    def process(self):
        if not self.pipeline.options.se_t1_maps:
            self.no_data("No path to additional SE T1 maps given")
        else:
            globexpr = os.path.join(
                self.pipeline.options.se_t1_maps, 
                "%s_*.nii.gz" % self.pipeline.options.subjid
            )
            maps = glob.glob(globexpr)
            if not maps:
                LOG.info(f" - No SE T1 maps for {self.pipeline.options.subjid} in {globexpr}")
            else:
                if len(maps) > 1:
                    LOG.warn(f" - Multiple SE T1 maps found for {self.pipeline.options.subjid}: {maps} - using first")
                map = ImageFile(maps[0])
                map.save(self.outfile("se_t1.nii.gz"))

class AdcMap(Module):
    def __init__(self):
        Module.__init__(self, "adc")

    def process(self):
        if not self.pipeline.options.adc_maps:
            self.no_data("No path to additional ADC maps given")
        else:
            globexpr = os.path.join(
                self.pipeline.options.adc_maps, 
                "%s_*.nii.gz" % self.pipeline.options.subjid
            )
            maps = glob.glob(globexpr)
            if not maps:
                LOG.info(f" - No ADC maps for {self.pipeline.options.subjid} in {globexpr}")
            else:
                if len(maps) > 1:
                    LOG.warn(f" - Multiple ADC maps found for {self.pipeline.options.subjid}: {maps} - using first")
                map = ImageFile(maps[0])
                map.save(self.outfile("adc.nii.gz"))

class Stats(StatsModule):
    def __init__(self):
        StatsModule.__init__(
            self, name="stats", 
            segs={
                "liver" : {
                    "dir" : "liver_seg_fix",
                    "glob" : "liver.nii.gz"
                },
                "spleen" : {
                    "dir" : "spleen_seg",
                    "glob" : "spleen.nii.gz"
                },
                "pancreas" : {
                    "dir" : "pancreas_seg_fix",
                    "glob" : "seg_pancreas.nii.gz"
                },
                "sat" : {
                    "dir" : "sat_seg_fix",
                    "glob" : "sat.nii.gz",
                    "params" : ["dummy"]  # Don't apply to any parameter maps
                },
                "kidney_cortex_l" : {
                    "dir" : "kidney_seg",
                    "glob" : "seg_kidney_cortex_l_t1.nii.gz"
                },
                "kidney_cortex_r" : {
                    "dir" : "kidney_seg",
                    "glob" : "seg_kidney_cortex_r_t1.nii.gz"
                },
                "kidney_medulla_l" : {
                    "dir" : "kidney_seg",
                    "glob" : "seg_kidney_medulla_l_t1.nii.gz"
                },
                "kidney_medulla_r" : {
                    "dir" : "kidney_seg",
                    "glob" : "seg_kidney_medulla_r_t1.nii.gz"
                },
            },
            params={
                "t2star" : {
                    "dir" : "t2star",
                    "glob" : "t2star.nii.gz",
                    "limits" : (2, 100),
                },
                "ff" : {
                    "dir" : "fat_fraction",
                    "glob" : "fat_fraction.nii.gz",
                    "limits" : (0, 1),
                },
                "t1_liver" : {
                    "dir" : "t1_liver",
                    "glob" : "t1_map.nii.gz",
                },
                "t1_kidney" : {
                    "dir" : "t1_kidney",
                    "glob" : "t1_map.nii.gz",
                },
                "b0" : {
                    "dir" : "b0",
                    "glob" : "b0.nii.gz",
                },
                "b1" : {
                    "dir" : "b1",
                    "glob" : "b1.nii.gz",
                },
                "t2" : {
                    "dir" : "t2",
                    "glob" : "t2.nii.gz",
                },
                "se_t1" : {
                    "dir" : "se_t1",
                    "glob" : "se_t1.nii.gz",
                },
                "adc" : {
                    "dir" : "adc",
                    "glob" : "adc.nii.gz",
                },
            },
            stats=["iqmean", "median", "iqstd", "mode", "fwhm"],
            seg_volumes=True,
        )

MODULES = [
    LiverSeg(),
    SpleenSeg(),
    KidneySeg(),
    PancreasSeg(),
    SatSeg(),
    KidneySegClean(),
    PancreasSegFix(),
    LiverSegFix(),
    SatSegFix(),
    T2Star(),
    FatFraction(),
    B0(),
    B1(),
    T2(),
    T1Liver(),
    T1Kidney(),
    SeT1Map(),
    AdcMap(),
    Stats(),
]

class ResusProcArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "resus_proc", __version__)
        self.add_argument("--t1-model", help="Filename or URL for T1 segmentation model weights", default="/spmstore/project/RenalMRI/trained_models/kidney_t1_molli_min_max.pt")
        self.add_argument("--pancreas-masks", help="Directory containing manual pancreas masks")
        self.add_argument("--liver-masks", help="Directory containing manual liver masks")
        self.add_argument("--sat-masks", help="Directory containing manual SAT masks")
        self.add_argument("--se-t1-maps", help="Directory containing additional SE T1 maps")
        self.add_argument("--adc-maps", help="Directory containing additional ADC maps")

class ResusProc(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "resus_proc", __version__, ResusProcArgumentParser(), MODULES)

if __name__ == "__main__":
    ResusProc().run()
