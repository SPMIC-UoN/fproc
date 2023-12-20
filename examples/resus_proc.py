import logging

from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, StatsModule

import numpy as np

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
 
class Stats(StatsModule):
    def __init__(self):
        StatsModule.__init__(
            self, name="stats", 
            segs={
                "liver" : {
                    "dir" : "liver_seg",
                    "glob" : "liver.nii.gz"
                },
                "spleen" : {
                    "dir" : "spleen_seg",
                    "glob" : "spleen.nii.gz"
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
                }
            },
            stats=["iqmean", "median", "iqstd"],
            seg_volumes=True,
        )

MODULES = [
    LiverSeg(),
    SpleenSeg(),
    T2Star(),
    FatFraction(),
    Stats(),
]

class ResusProcArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "resus_proc", __version__)
        
class ResusProc(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "resus_proc", __version__, ResusProcArgumentParser(), MODULES)

if __name__ == "__main__":
    ResusProc().run()
