import logging
import os

import nibabel as nib
import numpy as np
import radiomics

from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class Nifti(Module):
    def __init__(self):
        Module.__init__(self, "nifti")

    def process(self):
        datadir = self.pipeline.options.input
        t1 = os.path.join(datadir, "T1_asc3p2p_t1ad.img")
        if not os.path.exists(t1):
            t1 = os.path.join(datadir, "T1_asc_d3p2p_t1ad.img")
        nii_t1 = nib.load(t1)
        LOG.info(f" - T1 data from {t1}: {nii_t1.shape}")
        nii_t1 = nib.Nifti1Image(nii_t1.get_fdata(), nii_t1.header.get_best_affine(), nii_t1.header)
        nii_t1.to_filename(self.outfile("t1.nii.gz"))

        mask = os.path.join(datadir, "T1_asc_mask.img")
        nii_mask = nib.load(mask)
        LOG.info(f" - mask data from {mask}: {nii_mask.shape}")
        # Sometimes the mask is not in the right image space so force it to the same as the T1
        nii_mask = nib.Nifti1Image((nii_mask.get_fdata() > 0).astype(np.int32), nii_t1.header.get_best_affine(), nii_t1.header)
        nii_mask.to_filename(self.outfile("mask.nii.gz"))

class PyRadiomics(Module):
    def __init__(self):
        Module.__init__(self, "pyradiomics")

    def process(self):
        t1 = self.inimg("nifti", "t1.nii.gz", is_depfile=True)
        mask = self.inimg("nifti", "mask.nii.gz", is_depfile=True)
        mask_restricted = np.copy(mask.data)
        mask_restricted[t1.data < 200] = 0
        mask_restricted[t1.data > 1200] = 0
        mask.save_derived(mask_restricted, self.outfile("mask_restricted.nii.gz"))

        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllImageTypes()
        results = extractor.execute(t1.fpath, self.outfile("mask_restricted.nii.gz"))
        with open(self.outfile("radiomics_features.csv"), "w") as f:
            for k, v in results.items():
                if k.startswith("diagnostics"):
                    continue
                if "shape" in k:
                    continue
                f.write(f"{k},{v}\n")

MODULES = [
    Nifti(),
    PyRadiomics()
]

class DemistifiPyradiomicsArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "demistifi_pyradiomics", __version__)
        
class DemistifiPyradiomics(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "demistifi_pyradiomics", __version__, DemistifiPyradiomicsArgumentParser(), MODULES)

if __name__ == "__main__":
    DemistifiPyradiomics().run()
