"""
FPROC: Modules for generating parameter maps from raw data
"""
import logging

import numpy as np
import scipy

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


class MTR(Module):
    def __init__(self):
        Module.__init__(self, "mtr")

    def process(self):
        ondata = self.inimg("mtr", "mtr_on.nii.gz", check=False)
        offdata = self.inimg("mtr", "mtr_off.nii.gz", check=False)
        if ondata is not None and offdata is not None:
            LOG.info(f" - Using MTR ON/OFF data from {ondata.fname}, {offdata.fname}")
            off_on_data = np.stack([offdata.data, ondata.data], axis=-1)
            srcfile = ondata
        else:
            onoffdata = self.inimg("mtr", "mtr_on_off.nii.gz", check=False)
            if onoffdata is None:
                self.no_data("No MTR on/off data found")
            LOG.info(f" - Using MTR ON/OFF data from {onoffdata.fname}")
            off_on_data = np.flip(onoffdata.data, axis=-1)
            srcfile = onoffdata

        from ukat.mapping.mtr import MTR
        mapper = MTR(off_on_data, affine=srcfile.affine)
        srcfile.save_derived(mapper.mtr_map, self.outfile("mtr.nii.gz"))


class T2star(Module):
    def __init__(self, name="t2star", echos_glob="t2star_e_*.nii.gz", expected_echos=None):
        Module.__init__(self, name)
        self._echos_glob = echos_glob
        self._expected_echos = expected_echos

    def process(self):
        echos = self.inimgs("t2star", self._echos_glob)
        if self._expected_echos and len(echos) != self._expected_echos:
            self.bad_data(f"Expected {self._expected_echos} echos, got {len(echos)}")

        echos.sort(key=lambda x: x.EchoTime)
        imgdata = [e.data for e in echos]
        tes = np.array([1000*e.EchoTime for e in echos])
        LOG.info(f"TEs: {tes}")
        last_echo = echos[-1]
        affine = last_echo.affine
        last_echo.save_derived(last_echo.data, self.outfile("last_echo.nii.gz"))  

        voxel_sizes = last_echo.nii.header.get_zooms()
        resample_voxel_size = self.pipeline.options.t2star_resample
        if resample_voxel_size > 0:
            voxel_sizes = [round(v, 1) for v in voxel_sizes][:2]
            std_sizes = [round(resample_voxel_size, 1) for v in voxel_sizes][:2]
            if not np.allclose(voxel_sizes, std_sizes):
                LOG.info(f"T2* data has resolution: {voxel_sizes} - resampling to {std_sizes}")
                zoom_factors = [voxel_sizes[d] / std_sizes[d] for d in range(2)] + [1.0] * (last_echo.ndim - 2)
                imgdata = [scipy.ndimage.zoom(d, zoom_factors) for d in imgdata]
                # Need to scale affine by same amount so FOV is unchanged. Note that this means the corner
                # voxels will have the same co-ordinates in original and rescaled data. The FOV depicted
                # in fsleyes etc will be slightly smaller because the voxels are smaller and the co-ordinates
                # are defined as being the centre of the voxel
                for j in range(2):
                    affine[:, j] = affine[:, j] / zoom_factors[j]

        methods = [self.pipeline.options.t2star_method]
        if methods == ["all"]:
            methods = ["loglin", "2p_exp"]
        for method in methods:
            from ukat.mapping.t2star import T2Star
            mapper = T2Star(np.stack(imgdata, axis=-1), tes, affine=affine, method=method, multithread=False)
            last_echo.save_derived(mapper.t2star_map, self.outfile(f"t2star_{method}.nii.gz"))
            last_echo.save_derived(self._fix_r2star_units(mapper.r2star_map()), self.outfile(f"r2star_{method}.nii.gz"))
            last_echo.save_derived(mapper.m0_map, self.outfile(f"m0_{method}.nii.gz"))

    def _fix_r2star_units(self, r2star_data):    
        """
        Change R2* from ms^-1 to s^-1
        
        FIXME do we need to check current range?
        """
        return 1000.0*r2star_data


class T2(Module):
    def __init__(self, t2_dir="t2", t2_glob="t2_e*.nii.gz", echos=10, max_echos=11, methods=["exp", "stim"]):
        Module.__init__(self, "t2")
        self._dir = t2_dir
        self._glob = t2_glob
        self._echos = echos
        self._max_echos = max_echos
        self._methods = methods

    def process(self):
        echos = self.inimgs(self._dir, self._glob)
        echos.sort(key=lambda x: x.EchoTime)

        if not echos:
            self.no_data("No T2 mapping data found")
        num_echos = len(echos)
        if num_echos < self._echos or num_echos > self._max_echos:
            self.bad_data(f"Incorrect number of T2 echos found: {num_echos}, expected between {self._echos} and {self._max_echos}")
        if len(echos) > self._echos:
            LOG.warn(f"{num_echos} echos found - discarding last {num_echos - self._echos} echos")
            echos = echos[:self._echos]

        first_echo = echos[0]
        data = np.stack([i.data for i in echos], axis=-1)
        tes = np.array([i.echotime * 1000 for i in echos]) # In milliseconds
        for method in self._methods:
            LOG.info(f" - Doing T2 mapping {method.upper()} fit on data shape {data.shape}, TES {tes}, vendor: {first_echo.vendor.lower()}")
            if method.lower() == "exp":
                from ukat.mapping.t2 import T2
                mapper = T2(data, tes, first_echo.affine)
            elif method.lower() == "stim":
                from ukat.mapping.t2_stimfit import T2StimFit, StimFitModel
                model = StimFitModel(ukrin_vendor=first_echo.vendor.lower())
                mapper = T2StimFit(data, first_echo.affine, model)
            else:
                raise ValueError(f"Unrecognized method: {method}")

            LOG.info(f" - DONE T2 mapping {method.upper()} fit - saving")
            first_echo.save_derived(mapper.t2_map, self.outfile(f"t2_{method.lower()}.nii.gz"))
            first_echo.save_derived(mapper.m0_map, self.outfile(f"m0_{method.lower()}.nii.gz"))
            if hasattr(mapper, "r2"):
                first_echo.save_derived(mapper.r2, self.outfile(f"r2_{method.lower()}.nii.gz"))
            if hasattr(mapper, "r2_map"):
                first_echo.save_derived(mapper.r2_map, self.outfile(f"r2_{method.lower()}.nii.gz"))
            if hasattr(mapper, "b1_map"):
               first_echo.save_derived(mapper.b1_map, self.outfile(f"b1_{method.lower()}.nii.gz"))
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
        first_echo.save_derived(mapper.b1_map, self.outfile("b1_map.nii.gz"))
        LOG.info(f" - Saved data")

class T1Molli(Module):
    def __init__(self, name="t1_molli", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        molli_dir = self.kwargs.get("molli_dir", "molli_raw")
        molli_glob = self.kwargs.get("molli_glob", "molli_raw*.nii.gz")
        tis = self.kwargs.get("tis", [])
        tss = self.kwargs.get("tss", 0.0)
        if not tis:
            self.no_data("TIs not specified - cannot do T1 MOLLI fit")
        if any([ti for ti in tis if ti < 10]):
            tis = [ti * 1000 for ti in tis]
            LOG.warn(f"Looks like TIs were specified in seconds - converting to ms")
        LOG.info(f" - Found {len(tis)} TIs (ms): {tis}")
        LOG.info(f" - Using temporal slice spaceing: {tss}")

        imgs = self.inimgs(molli_dir, molli_glob)
        if not imgs:
            self.no_data("No raw MOLLI data sets found")

        for img in imgs:
            LOG.info(f" - Processing MOLLI data from {img.fname}")
            if img.nvols < len(tis):
                LOG.warn(f"Not enough volumes in raw MOLLI data for provided TIs ({img.nvols} vs {len(tis)}) - ignoring")
                continue
            elif img.nvols != len(tis):
                LOG.warn(f"{img.nvols} volumes in raw MOLLI data, only using first {len(tis)} volumes")

            from ukat.mapping.t1 import T1, magnitude_correct
            mapper = T1(img.data[..., :len(tis)], np.array(tis), img.affine, parameters=2, tss=tss)
            mapper.to_nifti(self.outdir, base_file_name=img.fname_noext)

class FatFractionDixon(Module):
    def __init__(self, name="fat_fraction", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        ff_name = self.kwargs.get("ff_name", "fat_fraction")
        fat = self.inimg(dixon_dir, "fat.nii.gz")
        water = self.inimg(dixon_dir, "water.nii.gz")
        ff_orig = self.inimg(dixon_dir, f"{ff_name}.nii.gz", check=False)
        if ff_orig is not None:
            LOG.info(f" - Saving fat fraction map from DIXON: {ff_orig.fname} -> {ff_name}_orig")
            ff_orig.save(self.outfile(f"{ff_name}_orig.nii.gz"))

        water_data = self.resample(water, fat, allow_rotated=True).get_fdata()
        ff = np.zeros_like(fat.data, dtype=np.float32)
        valid = fat.data + water_data > 0
        ff[valid] = fat.data.astype(np.float32)[valid] / (fat.data + water_data)[valid]
        LOG.info(f" - Saving fat/water derived fat fraction map as {ff_name}_der")
        fat.save_derived(ff, self.outfile(f"{ff_name}_der.nii.gz"))

class T2starDixon(Module):
    def __init__(self, name="t2star_dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        t2star_name = self.kwargs.get("t2star_name", "t2star")
        imgs = self.copyinput(dixon_dir, f"{t2star_name}.nii.gz")
        if imgs:
            # 100 is a fill value - replace with something easier to exclude in stats
            LOG.info(" - Saving T2* map with excluded fill value")
            exclude_fill = np.copy(imgs[0].data)
            exclude_fill[np.isclose(exclude_fill, 100)] = -9999
            imgs[0].save_derived(exclude_fill, self.outfile(f"{t2star_name}_exclude_fill.nii.gz"))

class B1(Module):
    def __init__(self, name="b1", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        b1_dir = self.kwargs.get("b1_dir", "b1")
        b1_glob = self.kwargs.get("b1_glob", "b1.nii.gz")
        imgs = self.inimgs(b1_dir, b1_glob, src=self.INPUT)
        for img in imgs:
            LOG.info(f" - Saving B1 map from {img.fname}")
            img.save(self.outfile(img.fname))

            import ukat
            b1scaled = ukat.utils.tools.rescale_b1_map(img.data)
            LOG.info(f" - Saving rescaled B1 map from {img.fname}")
            img.save_derived(b1scaled, self.outfile(img.fname.replace(".nii", "_rescaled.nii")))
