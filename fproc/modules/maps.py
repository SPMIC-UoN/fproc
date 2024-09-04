"""
FPROC: Modules for generating parameter maps from raw data
"""
import logging

import numpy as np

from fproc.module import Module

LOG = logging.getLogger(__name__)

class B0(Module):
    def __init__(self):
        Module.__init__(self, "b0")

    def process(self):
        phasedata, tes = [], []
        phasemaps = self.inimgs("b0", "b0_phase_echo*.nii.gz")
        if not phasemaps:
            realmaps = self.inimgs("b0", "b0_real_echo*.nii.gz")
            imagmaps = self.inimgs("b0", "b0_imag_echo*.nii.gz")
            if not realmaps:
                self.no_data("No phase or real part B0 maps")
            if not imagmaps:
                self.no_data("No phase or imaginary part B0 maps")
            if len(realmaps) != len(imagmaps):
                self.bad_data("Different number of real and imaginary maps")

            for real, imag in zip(realmaps, imagmaps):
                if real.echotime != imag.echotime:
                    self.bad_data(f"Real and imaginary maps {real.fname}, {imag.fname} do not have the same echo time: {real.echotime} vs {imag.echotime}")
                LOG.info(f" - Calculating phase from real/imag: {real.fname}, {imag.fname}, TE: {real.echotime}")
                data_phase = -np.arctan2(imag.data, real.data)
                phasedata.append(data_phase)
                real.save_derived(real.data, self.outfile(real.fname.replace("_real", "_realcopy")))
                real.save_derived(imag.data, self.outfile(real.fname.replace("_real", "_imagcopy")))
                real.save_derived(data_phase, self.outfile(real.fname.replace("_real", "_phase")))
                tes.append(real.echotime * 1000)
                srcfile = real
        else:
            for f in phasemaps:
                LOG.info(f" - Found phase data: {f.fname}, TE: {f.echotime}")
                phasedata.append(f.data)
                tes.append(f.echotime * 1000)
                srcfile = f

        if len(phasedata) != 2:
            LOG.warn(" - More than two echos found - using first two only")
            phasedata = phasedata[:2]
            tes = tes[:2]

        stacked_data = np.stack(phasedata, axis=-1)
        from ukat.mapping.b0 import B0
        mapper = B0(stacked_data, tes, affine=srcfile.affine)

        # Save output maps to Nifti
        srcfile.save_derived(mapper.b0_map, self.outfile("b0.nii.gz"))
        srcfile.save_derived(mapper.phase0, self.outfile("b0phase0.nii.gz"))
        srcfile.save_derived(mapper.phase1, self.outfile("b0phase1.nii.gz"))
        srcfile.save_derived(mapper.phase_difference, self.outfile("b0phasediff.nii.gz"))

