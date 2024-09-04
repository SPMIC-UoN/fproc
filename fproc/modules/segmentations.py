"""
FPROC: Segmentations of various body parts
"""
import logging
import validators
import wget

from fsort import ImageFile
from fproc.module import Module

LOG = logging.getLogger(__name__)

class LiverDixon(Module):
    def __init__(self):
        Module.__init__(self, "seg_liver_dixon")

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
            logfile=f'seg.log'
        )

        seg = self.inimg("seg_liver_dixon", "liver.nii.gz", src=self.OUTPUT)
        water = self.inimg("dixon", "water.nii.gz")
        self.lightbox(water, seg, name="liver_water_lightbox", tight=True)

class SpleenDixon(Module):
    def __init__(self):
        Module.__init__(self, "seg_spleen_dixon")

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
            logfile=f'seg.log'
        )

        seg = self.inimg("seg_spleen_dixon", "spleen.nii.gz", src=self.OUTPUT)
        water = self.inimg("dixon", "water.nii.gz")
        self.lightbox(water, seg, name="spleen_water_lightbox", tight=True)

class PancreasEthrive(Module):
    def __init__(self):
        Module.__init__(self, "seg_pancreas_ethrive")

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
            logfile=f'seg.log'
        )

        seg = self.inimg("seg_pancreas_ethrive", "pancreas.nii.gz", src=self.OUTPUT)
        self.lightbox(ethrive, seg, name="pancreas_ethrive_lightbox", tight=True)

class KidneyT1(Module):
    def __init__(self):
        Module.__init__(self, "seg_kidney_t1")

    def process(self):
        t1_map = self.inimg("molli_kidney", "t1_map.nii.gz")
        LOG.info(f" - Segmenting KIDNEY using T1 data: {t1_map.fname}")
        self.runcmd([
            'kidney_t1_seg',
            '--input', t1_map.dirname,
            '--subjid', '',
            '--display-id', self.pipeline.options.subjid,
            '--t1', t1_map.fname,
            '--model', self.pipeline.options.kidney_t1_model,
            '--noclean',
            '--output', self.outdir,
            '--outprefix', f'kidney'],
            logfile=f'seg.log'
        )

        seg = self.inimg("seg_kidney_t1", "kidney_all_t1.nii.gz", src=self.OUTPUT)
        self.lightbox(t1_map, seg, name="kidney_t1_lightbox", tight=True)

class KidneyT2w(Module):
    def __init__(self, t2w_srcdir="t2w"):
        Module.__init__(self, "seg_kidney_t2w")
        self.t2w_srcdir = t2w_srcdir

    def process(self):
        t2w_map = self.inimg(self.t2w_srcdir, "t2w.nii.gz")
        LOG.info(f" - Segmenting KIDNEY using T2w data: {t2w_map.fname}")

        model_weights = self.pipeline.options.kidney_t2w_model
        if validators.url(model_weights):
            # Download weights from supplied URL
            LOG.info(f" - Downloading model weights from {model_weights}")
            wget.download(model_weights, "model.h5")
            model_weights = "model.h5"
        else:
            LOG.info(f" - Using model weights from {model_weights}")

        # The segmentor needs both the image array and affine so the size of each voxel is known. post_process=True removes all but
        # the largest two areas in the mask e.g. removes small areas of incorrectly categorised tissue. This can cause issues if the
        # subject has more or less than two kidneys though.
        from ukat.segmentation import whole_kidney
        segmentation = whole_kidney.Segmentation(t2w_map.data, t2w_map.affine, post_process=True, binary=True, weights=model_weights)
        segmentation.to_nifti(output_directory=self.outdir, base_file_name=f"kidney", maps=['mask', 'left', 'right', 'individual'])

        # Generate overlay images using T2w map
        LOG.info(f" - Generating overlay image for T2w segmentation using {t2w_map.fname}")
        mask_img = ImageFile(self.outfile("kidney_mask.nii.gz"), warn_json=False)
        self.lightbox(t2w_map, mask_img, "kidney_t2w_lightbox")

class SatDixon(Module):
    def __init__(self):
        Module.__init__(self, "seg_sat_dixon")

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

        seg = self.inimg("seg_sat_dixon", "sat.nii.gz", src=self.OUTPUT)
        water = self.inimg("dixon", "water.nii.gz")
        self.lightbox(water, seg, name="sat_water_lightbox", tight=True)

class LegDixon(Module):
    def __init__(self, dixon_srcdir="dixon"):
        Module.__init__(self, "seg_leg_dixon")
        self.dixon_srcdir = dixon_srcdir

    def process(self):
        water = self.inimg(self.dixon_srcdir, "water.nii.gz")
        fat = self.inimg(self.dixon_srcdir, "water.nii.gz")
        LOG.info(f" - Segmenting LEG using water: {water.fpath}, fat: {fat.fpath}")
        self.runcmd([
            'leg_dixon_seg',
            '--water', water.fpath,
            '--fat', fat.fpath,
            '--model', self.pipeline.options.leg_dixon_model,
            '--output', self.outfile("leg.nii.gz")
        ], logfile=f'seg.log')

        seg = self.inimg("seg_leg_dixon", "leg.nii.gz", src=self.OUTPUT)
        self.lightbox(water, seg, name="leg_water_lightbox", tight=True)
