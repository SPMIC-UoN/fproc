import logging

from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, StatsModule

import numpy as np

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class T2SimpleExp(Module):
    def __init__(self):
        Module.__init__(self, "t2_exp")

    def process(self):
        imgs = self.inimgs("t2", "t2_e*.nii.gz")
        if not imgs:
            self.no_data("No T2 mapping data found")
        elif len(imgs) not in (10, 11):
            self.bad_data(f"Incorrect number of T2 echos found: {len(imgs)}, expected 10 or 11")
        if len(imgs) == 11:
            LOG.warn("11 echos found - discarding last echo")
            imgs = imgs[:10]

        # Do this to make sure we get the echos in the correct order!
        imgs = [self.inimg("t2", f"t2_e_{echo}.nii.gz") for echo in range(1, 11)]

        # Import is expensive so delay until we need it
        from ukat.mapping.t2 import T2
        tes = np.array([i.echotime * 1000 for i in imgs]) # In milliseconds
        data = np.stack([i.data for i in imgs], axis=-1)
        first_echo = imgs[0]
        LOG.info(f" - Doing T2 mapping EXP fit on data shape {data.shape}, TEs: {tes}")
        mapper = T2(data, tes, first_echo.affine)
        LOG.info(f" - DONE T2 mapping EXP fit - saving")
        first_echo.save_derived(mapper.t2_map, self.outfile("t2_map.nii.gz"))
        first_echo.save_derived(mapper.m0_map, self.outfile("m0_map.nii.gz"))
        first_echo.save_derived(mapper.r2, self.outfile("r2_map.nii.gz"))
        LOG.info(f" - Saved data")

class T2Stim(Module):
    def __init__(self):
        Module.__init__(self, "t2_stim")

    def process(self):
        imgs = self.inimgs("t2", "t2_e*.nii.gz")
        if not imgs:
            self.no_data("No T2 mapping data found")
        elif len(imgs) not in (10, 11):
            self.bad_data(f"Incorrect number of T2 echos found: {len(imgs)}, expected 10 or 11")

        if len(imgs) == 11:
            LOG.warn("11 echos found - discarding last echo")
            imgs = imgs[:10]

        # Do this to make sure we get the echos in the correct order!
        imgs = [self.inimg("t2", f"t2_e_{echo}.nii.gz") for echo in range(1, 11)]

        # Import is expensive so delay until we need it
        from ukat.mapping.t2_stimfit import T2StimFit, StimFitModel
        first_echo = imgs[0]
        model = StimFitModel(ukrin_vendor=first_echo.vendor.lower())
        #tes = np.array([i.echotime for i in imgs])
        data = np.stack([i.data for i in imgs], axis=-1)
        LOG.info(f" - Doing T2 mapping STIM fit on data shape {data.shape}, vendor: {first_echo.vendor.lower()}")
        mapper = T2StimFit(data, first_echo.affine, model)
        LOG.info(f" - DONE T2 mapping STIM fit - saving")
        first_echo.save_derived(mapper.t2_map, self.outfile("t2_map.nii.gz"))
        first_echo.save_derived(mapper.m0_map, self.outfile("m0_map.nii.gz"))
        first_echo.save_derived(mapper.r2_map, self.outfile("r2_map.nii.gz"))
        LOG.info(f" - Saved data")

MODULES = [
    T2SimpleExp(),
    T2Stim(),
]

class Tk21T2ArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "tk21_t2", __version__)
        
class Tk21T2(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "tk21_t2", __version__, Tk21T2ArgumentParser(), MODULES)

if __name__ == "__main__":
    Tk21T2().run()
