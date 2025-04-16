"""
FPROC: Modules for generating parameter maps from raw data
"""
import logging
import os
import glob

import numpy as np
import scipy

from fproc.module import Module
from fsort.image_file import ImageFile

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
    def __init__(self, name="t2star", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t2star_dir = self.kwargs.get("t2star_dir", "t2star")
        echos_glob = self.kwargs.get("echos_glob", "t2star_e_*.nii.gz")
        expected_echos = self.kwargs.get("expected_echos", None)
        echos = self.inimgs(t2star_dir, echos_glob)
        if not echos:
            self.no_data("No T2* mapping data found")
        elif expected_echos and len(echos) != expected_echos:
            self.bad_data(f"Expected {expected_echos} echos, got {len(echos)}")
        else:
            LOG.info(f" - Found {len(echos)} echos")

        echos.sort(key=lambda x: x.EchoTime)
        imgdata = [e.data for e in echos]
        tes = np.array([e.EchoTime for e in echos])
        if all(tes < 1):
            LOG.info(f" - Looks like TEs were specified in seconds - converting")
            tes = tes * 1000
        LOG.info(f" - TEs: {tes}")
        last_echo = echos[-1]
        affine = last_echo.affine
        last_echo.save_derived(last_echo.data, self.outfile("last_echo.nii.gz"))  

        voxel_sizes = last_echo.nii.header.get_zooms()
        resample_voxel_size = self.pipeline.options.t2star_resample
        if resample_voxel_size > 0:
            voxel_sizes = [round(v, 1) for v in voxel_sizes][:2]
            std_sizes = [round(resample_voxel_size, 1) for v in voxel_sizes][:2]
            if not np.allclose(voxel_sizes, std_sizes):
                LOG.info(f" - T2* data has resolution: {voxel_sizes} - resampling to {std_sizes}")
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
        molli_src = self.kwargs.get("molli_src", self.INPUT)
        mdr = self.kwargs.get("mdr", False)
        tis = self.kwargs.get("tis", None)
        parameters = self.kwargs.get("parameters", 3)
        molli = self.kwargs.get("molli", True)
        tis_use_md = self.kwargs.get("tis_use_md", False)
        if tis is None or len(tis) == 0:
            LOG.info(" - TIs not specified - will try to read from metadata")
            tis = None
        else:
            if tis_use_md:
                LOG.info(" - Default set of TIs specified - but will use metadata if available")
            if any([ti for ti in tis if ti < 10]):
                tis = [ti * 1000 for ti in tis]
                LOG.warn(f"Looks like TIs were specified in seconds - converting to ms")
            LOG.info(f" - Found {len(tis)} TIs (ms): {tis}")
        tss = self.kwargs.get("tss", 0.0)
        LOG.info(f" - Using temporal slice spaceing: {tss}")

        imgs = self.inimgs(molli_dir, molli_glob, src=molli_src)
        if imgs:
            for img in imgs:
                LOG.info(f" - Processing MOLLI data from {img.fname} using MDR={mdr}, MOLLI corrections={molli}, parameters={parameters}")
                if tis_use_md or tis is None:
                    # Consider TIs to be equivalent if they are with 10ms of each other
                    img_tis = np.array(np.unique([int(t/10) for t in img.inversiontimedelay if int(t) > 0])) * 10
                    if len(img_tis) == 0 and tis is None:
                        LOG.warn(" - No TIs found in metadata and no default provided - skipping this image")
                        continue
                    elif len(img_tis) == 0:
                        LOG.warn(" - No TIs found in metadata - using default provided")
                        img_tis = tis
                    else:
                        LOG.info(f" - Found {len(img_tis)} TIs (ms): {img_tis} in metadata")
                else:
                    img_tis = tis

                if img.nvols < len(img_tis):
                    LOG.warn(f"Not enough volumes in raw MOLLI data for provided TIs ({img.nvols} vs {len(img_tis)}) - ignoring")
                    continue
                elif img.nvols != len(img_tis):
                    LOG.warn(f"{img.nvols} volumes in raw MOLLI data, only using first {len(img_tis)} volumes")
                from ukat.mapping.t1 import T1
                mapper = T1(img.data[..., :len(img_tis)], np.array(img_tis), img.affine, parameters=parameters, tss=tss, molli=molli, mdr=mdr)
                mapper.to_nifti(self.outdir, base_file_name=img.fname_noext)
                if mdr:
                    # Save out registered data
                    reg_data = mapper.pixel_array
                    img.save_derived(reg_data, self.outfile(img.fname.replace(".nii.gz", "_reg.nii.gz")))

                t1_map_data = np.copy(mapper.t1_map)
                r2_thresh = self.kwargs.get("r2_thresh", 0.0)
                if r2_thresh > 0:
                    LOG.info(f" - Applying R2 threshold of {r2_thresh} to T1 map")
                    r2_map = self.single_inimg(self.name, img.fname_noext + "_r2.nii.gz", src=self.OUTPUT)
                    t1_map_data[np.abs(r2_map.data) < r2_thresh] = 0

                t1_thresh = self.kwargs.get("t1_thresh", None)
                if t1_thresh is not None:
                    LOG.info(f" - Applying threshold of {t1_thresh} to T1 map")
                    if isinstance(t1_thresh, (int, float)):
                        t1_thresh = (None, t1_thresh)
                    t1_min, t1_max = tuple(t1_thresh)
                    thresh_mode = self.kwargs.get("t1_thresh_mode", "zero")
                    if thresh_mode == "zero":
                        t1_map_data[t1_map_data > t1_max] = 0
                        t1_map_data[t1_map_data < t1_min] = 0
                    elif thresh_mode == "clip":
                        t1_map_data[t1_map_data > t1_max] = t1_max
                        t1_map_data[t1_map_data < t1_min] = t1_min
                    else:
                        raise ValueError(f"Unrecognized threshold mode: {thresh_mode}")
                fname = img.fname.replace(".nii.gz", "_t1_map.nii.gz")
                img.save_derived(t1_map_data, self.outfile(fname))
                LOG.info(f" - Final T1 map saved to {fname}")

        elif self.kwargs.get("use_scanner_maps", True):
            LOG.info(f" - No raw MOLLI data found in {molli_dir}/{molli_glob} - looking for scanner T1 map/confidence images")
            map_glob = self.kwargs.get("map_glob", "t1_map*.nii.gz")
            conf_glob = self.kwargs.get("conf_glob", "t1_conf*.nii.gz")
            if self.inimgs(molli_dir, map_glob):
                LOG.info(f" - Copying T1 map/confidence images")
                self.copyinput(molli_dir, map_glob)
                self.copyinput(molli_dir, conf_glob)
            else:
                self.no_data(f"No raw MOLLI found and no scanner computer T1 maps in {molli_dir}/{map_glob}")
        else:
            self.no_data(f"No raw MOLLI data found in {molli_dir}/{molli_glob}")

        imgs = self.inimgs(self.name, "*t1_conf*.nii.gz", src=self.OUTPUT)
        if not imgs:
            LOG.info(f" - No T1 conf map - copying T1 maps to T1 conf")
            imgs = self.inimgs(self.name, "*t1_map*.nii.gz", src=self.OUTPUT)
            for img in imgs:
                img.save(self.outfile(img.fname.replace("map", "conf")))


class T1SE(Module):
    def __init__(self, name="t1_se", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        se_dir = self.kwargs.get("se_dir", "t1_se")
        se_mag_glob = self.kwargs.get("se_mag_glob", "t1_se_mag*.nii.gz")
        se_ph_glob = self.kwargs.get("se_ph_glob", "t1_se_ph*.nii.gz")
        se_src = self.kwargs.get("se_src", self.INPUT)
        parameters = self.kwargs.get("parameters", 2)
        mag_only = self.kwargs.get("mag_only", True)
        mdr = self.kwargs.get("mdr", False)
        tis = self.kwargs.get("tis", None)
        if tis is None or len(tis) == 0:
            LOG.info(" - TIs not specified - will try to read from metadata")
        else:
            if any([ti for ti in tis if ti < 10]):
                tis = [ti * 1000 for ti in tis]
                LOG.warn(f"Looks like TIs were specified in seconds - converting to ms")
            LOG.info(f" - Supplied {len(tis)} TIs (ms): {tis}")
        tss = self.kwargs.get("tss", 0.0)
        LOG.info(f" - Using temporal slice spaceing: {tss}")
        LOG.info(f" - Using {parameters}-parameter fit")

        mag_imgs = self.inimgs(se_dir, se_mag_glob, src=se_src)
        if not mag_only:
            ph_imgs = self.inimgs(se_dir, se_ph_glob, src=se_src)
            if len(mag_imgs) != len(ph_imgs):
                self.bad_data(f"Different number of magnitude and phase images: {len(mag_imgs)} vs {len(ph_imgs)}")
        else:
            ph_imgs = [None] * len(mag_imgs)

        if not mag_imgs:
            self.no_data(f"No T1 SE data found in {se_dir}/{se_mag_glob}")

        for mag, ph in zip(mag_imgs, ph_imgs):
            LOG.info(f" - Processing SE data from {mag.fname}")
            if tis is None or len(tis) == 0:
                img_tis = [float(t) for t in mag.inversiontimedelay if float(t) > 0]
                if len(img_tis) == 0:
                    LOG.warn(" - No TIs found in metadata - skipping this image")
                    continue
                else:
                    LOG.info(f" - Found {len(img_tis)} TIs (ms): {img_tis} in metadata")
            else:
                img_tis = tis

            if mag.nvols < len(img_tis):
                LOG.warn(f"Not enough volumes in magnitude data for provided TIs ({mag.nvols} vs {len(img_tis)}) - ignoring")
                continue
            elif mag.nvols != len(img_tis):
                LOG.warn(f"{mag.nvols} volumes in magnitude data, only using first {len(img_tis)} volumes")

            from ukat.mapping.t1 import T1, magnitude_correct
            from ukat.utils.tools import convert_to_pi_range
            if ph is not None:
                LOG.info(f" - Found phase data {ph.fname} - correcting magnitude data")
                if ph.nvols < len(img_tis):
                    LOG.warn(f"Not enough volumes in phase data for provided TIs ({ph.nvols} vs {len(img_tis)}) - ignoring")
                    continue
                elif ph.nvols != len(img_tis):
                    LOG.warn(f"{ph.nvols} volumes in phase data, only using first {len(img_tis)} volumes")

                phase_data = convert_to_pi_range(ph.data)
                complex_data = mag.data * (np.cos(phase_data) + 1j * np.sin(phase_data))
                magnitude_corrected = np.nan_to_num(magnitude_correct(complex_data))
            else:
                magnitude_corrected = mag.data

            acq_order = "centric" if mag.manufacturer.lower() == "siemens" else "ascend"
            LOG.info(f" - Using acquisition order: {acq_order} for vendor {mag.manufacturer}")
            mapper = T1(magnitude_corrected[..., :len(img_tis)], np.array(img_tis), mag.affine, tss=tss, mag_corr=not mag_only, parameters=parameters, mdr=mdr, acq_order=acq_order)
            mapper.to_nifti(self.outdir, base_file_name=mag.fname_noext.replace("_mag", ""))
            if mdr:
                # Save out registered data
                mag.save_derived(mapper.pixel_array, self.outfile(mag.fname.replace(".nii.gz", "_reg.nii.gz")))

            #r2_thresh = self.kwargs.get("r2_thresh", 0.0)
            #if r2_thresh > 0:
            #    LOG.info(f" - Applying R2 threshold of {r2_thresh} to T1 map")
            #    r2_map = self.single_inimg(self.name, img.fname_noext + "_r2.nii.gz", src=self.OUTPUT)
            #    t1_map_thresh = np.copy(mapper.t1_map)
            #    t1_map_thresh[np.abs(r2_map.data) < r2_thresh] = 0
            #    img.save_derived(t1_map_thresh, self.outfile(img.fname.replace(".nii.gz", "_r2_thresh.nii.gz")))

            #t1_thresh = self.kwargs.get("t1_thresh", None)
            #if t1_thresh is not None:
            #    LOG.info(f" - Applying T1 max threshold of {t1_thresh} to T1 map")
            #    t1_map_thresh = np.copy(mapper.t1_map)
            #    t1_map_thresh[t1_map_thresh > t1_thresh] = 0
            #    img.save_derived(t1_map_thresh, self.outfile(img.fname.replace(".nii.gz", "_t1_thresh.nii.gz")))


class FatFractionDixon(Module):
    def __init__(self, name="fat_fraction", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "dixon")
        ff_name = self.kwargs.get("ff_name", "fat_fraction")
        fat = self.inimg(dixon_dir, "fat.nii.gz")
        water = self.inimg(dixon_dir, "water.nii.gz")
        ff_scanner = self.inimg(dixon_dir, f"{ff_name}.nii.gz", check=False)

        if ff_scanner is not None:
            ff_data = ff_scanner.data
            ff_max = np.percentile(ff_data, 95)
            LOG.info(f" - Fat fraction 95% percentile: {ff_max}")
            if ff_max < 2:
                LOG.info(" - Fat fraction scaled 0-1 - saving as percentage")
                ff_data = ff_data * 100
            elif ff_max > 200:
                if ff_scanner.philipsscaleslope is not None:
                    LOG.info(f" - Found Philips enhanced scale slope - scaling Fat fraction by {ff_scanner.philipsscaleslope}")
                    ff_data = ff_data * ff_scanner.philipsscaleslope
                    if ff_max * ff_scanner.philipsscaleslope > 200 or ff_max * ff_scanner.philipsscaleslope < 2:
                        LOG.warn(f"Scaled fat fraction still not in expected range: {ff_max * ff_scanner.philipsscaleslope}")
                else:
                    LOG.warn("Fat fraction not in expected range and no scale slope found - check statistics")
            LOG.info(f" - Saving scanner fat fraction to {ff_name}_scanner.nii.gz")
            ff_scanner.save_derived(ff_data, self.outfile(f"{ff_name}_scanner.nii.gz"))
        else:
            LOG.info(" - Scanner derived fat fraction map not found")

        if fat is not None and water is not None:
            water_data = self.resample(water, fat, allow_rotated=True).get_fdata()
            ff = np.zeros_like(fat.data, dtype=np.float32)
            valid = fat.data + water_data > 0
            ff[valid] = fat.data.astype(np.float32)[valid] * 100 / (fat.data + water_data)[valid]
            LOG.info(f" - Saving fat/water derived fat fraction map as {ff_name}_calc")
            fat.save_derived(ff, self.outfile(f"{ff_name}_calc.nii.gz"))
        else:
            LOG.info(" - Could not find fat/water images - not calculating fat fraction")
            if ff_scanner is None:
                LOG.warn("No fat fraction data found")

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

class DixonDerived(Module):
    """
    Derived Dixon maps
    """
    def __init__(self, name="dixon", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dixon_dir = self.kwargs.get("dixon_dir", "../fsort/dixon")
        globs = self.kwargs.get("globs", ["fat.nii.gz", "water.nii.gz", "fat_fraction.nii.gz", "t2star.nii.gz", "ip.nii.gz", "op.nii.gz"])
        for img_glob in globs:
            imgs = self.inimgs(dixon_dir, img_glob, src=self.OUTPUT)
            if not imgs:
                LOG.info(f" - No Dixon data found in {dixon_dir}/{img_glob}")
                continue            
            for img in imgs:
                LOG.info(f" - Saving Dixon data from {img.fname}")
                img.save(self.outfile(img.fname))

        # T2* with excluded fill value
        t2star_name = self.kwargs.get("t2star_name", "t2star")
        img = self.single_inimg(dixon_dir, f"{t2star_name}.nii.gz", src=self.OUTPUT)
        if img is not None:
            # 100 is a fill value - replace with something easier to exclude in stats
            LOG.info(" - Saving T2* map with excluded fill value")
            exclude_fill = np.copy(img.data)
            exclude_fill[np.isclose(exclude_fill, 100)] = -9999
            img.save_derived(exclude_fill, self.outfile(f"{t2star_name}_exclude_fill.nii.gz"))

        # Scanner derived and calculated FF map
        ff_name = self.kwargs.get("ff_name", "fat_fraction")
        fat = self.single_inimg(dixon_dir, "fat.nii.gz")
        water = self.single_inimg(dixon_dir, "water.nii.gz")
        ff_scanner = self.single_inimg(dixon_dir, f"{ff_name}.nii.gz", src=self.OUTPUT)
        if ff_scanner is not None:
            ff_data = ff_scanner.data
            ff_max = np.percentile(ff_data, 95)
            LOG.info(f" - Fat fraction 95% percentile: {ff_max}")
            if ff_max < 2:
                LOG.info(" - Fat fraction scaled 0-1 - saving as percentage")
                ff_data = ff_data * 100
            elif ff_max > 200:
                if ff_scanner.philipsscaleslope is not None:
                    LOG.info(f" - Found Philips enhanced scale slope - scaling Fat fraction by {ff_scanner.philipsscaleslope}")
                    ff_data = ff_data * ff_scanner.philipsscaleslope
                    if ff_max * ff_scanner.philipsscaleslope > 200 or ff_max * ff_scanner.philipsscaleslope < 2:
                        LOG.warn(f"Scaled fat fraction still not in expected range: {ff_max * ff_scanner.philipsscaleslope}")
                else:
                    LOG.warn("Fat fraction not in expected range and no scale slope found - check statistics")
            LOG.info(f" - Saving scanner fat fraction to {ff_name}_scanner.nii.gz")
            ff_scanner.save_derived(ff_data, self.outfile(f"{ff_name}_scanner.nii.gz"))
        else:
            LOG.info(" - Scanner derived fat fraction map not found")

        if fat is not None and water is not None:
            water_data = self.resample(water, fat, allow_rotated=True).get_fdata()
            ff = np.zeros_like(fat.data, dtype=np.float32)
            valid = fat.data + water_data > 0
            ff[valid] = fat.data.astype(np.float32)[valid] * 100 / (fat.data + water_data)[valid]
            LOG.info(f" - Saving fat/water derived fat fraction map as {ff_name}_calc")
            fat.save_derived(ff, self.outfile(f"{ff_name}_calc.nii.gz"))
        else:
            LOG.info(" - Could not find fat/water images - not calculating fat fraction")
            if ff_scanner is None:
                LOG.warn("No fat fraction data found")

        # IP / OP if not scanner generated
        ip_glob = self.kwargs.get("ip_glob", "ip.nii.gz")
        op_glob = self.kwargs.get("op_glob", "op.nii.gz")
        ip = self.single_inimg(dixon_dir, ip_glob, src=self.OUTPUT, warn=False)
        op = self.single_inimg(dixon_dir, op_glob, src=self.OUTPUT, warn=False)
        if ip is not None:
            LOG.info(f" - Saving scanner generated IP map from {ip.fname}")
            ip.save(self.outfile("ip.nii.gz"))
        else:
            LOG.info(f" - No scanner generated IP map found - using fat + water")
            ip = fat.data + water.data
            fat.save_derived(ip, self.outfile("ip.nii.gz"))
        if op is not None:
            LOG.info(f" - Saving scanner generated OP map from {op.fname}")
            op.save(self.outfile("op.nii.gz"))
        else:
            LOG.info(f" - No scanner generated OP map found - using abs(water - fat)")
            op = np.abs(water.data - fat.data)
            fat.save_derived(op, self.outfile("op.nii.gz"))

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


class MapFix(Module):
    """
    Module that can replace automatic map with an external one
    """
    def __init__(self, map_dir, name=None, **kwargs):
        if name is None:
            name = map_dir + "_fix"
        Module.__init__(self, name, **kwargs)
        self._map_dir = map_dir

    def process(self):
        fix_dir_option = self.kwargs.get("fix_dir_option", self.name + "_fix")
        fix_dir = getattr(self.pipeline.options, fix_dir_option, None)
        try_to_fix = False
        if not fix_dir:
            LOG.warn(" - No fixed maps dir specified - will not try to fix maps")
        elif not os.path.exists(fix_dir):
            LOG.warn(f" - Fixed maps dir {fix_dir} does not exist - will not try to fix maps")
        else:
            LOG.info(f" - Looking for fixed maps in {fix_dir}")
            try_to_fix = True

        maps = self.kwargs.get("maps", {})
        for map_glob, fix_spec in maps.items():
            ignore_missing = False
            fname = None
            if isinstance(fix_spec, str):
                fix_glob = fix_spec
            else:
                fix_glob = fix_spec.get("glob", None)
                ignore_missing = fix_spec.get("ignore_missing", False)
                fname = fix_spec.get("fname", fname)
            map_img = self.single_inimg(self._map_dir, map_glob, src=self.kwargs.get("map_src", self.OUTPUT), warn=False)
            if map_img is None and ignore_missing:
                LOG.warn(f"No map found matching {self._map_dir}/{map_glob} - ignoring")
                continue
            elif map_img is None:
                LOG.info(f" - No original image matching {self._map_dir}/{map_glob} - will check for fix anyway")
            else:
                LOG.info(f" - Checking for fixed version of {map_img.fname}")
                if not fname:
                    fname = map_img.fname

            fixed_map = None
            if try_to_fix:
                globexpr = os.path.join(fix_dir, fix_glob % self.pipeline.options.subjid)
                fixed_maps = glob.glob(globexpr, recursive=True)
                if not fixed_maps:
                    LOG.info(f" - No fixed maps found in {globexpr}")
                else:
                    if len(fixed_maps) > 1:
                        LOG.warn(f" - Multiple matching 'fixed' maps found: {fixed_maps} - using first")
                    fixed_map = ImageFile(fixed_maps[0])
                    fixed_data = fixed_map.data
                    if not fname:
                        fname = fixed_map.fname

                    LOG.info(f" - Saving fixed map from {fname} as int")
                    fixed_map = fixed_map.save_derived(fixed_data, self.outfile(fname))

            if fixed_map is None and map_img is not None:
                LOG.info(f" - Saving original map from {map_img.fname} as {fname}")
                fixed_map = map_img
                fixed_map.save(self.outfile(fname))
            elif fixed_map is None:
                LOG.warn(f" - No fixed map found - will not save")

class AdditionalMap(Module):
    def __init__(self, name, **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        srcdir_option = self.kwargs.get("srcdir_option", self.name + "_dir")
        srcdir = getattr(self.pipeline.options, srcdir_option, None)
        if srcdir is None:
            LOG.info(" - No source directory specified - not checking for additional maps")
            return

        maps = self.kwargs.get("maps", {})
        for fname, glob in maps.items():
            if "%s" in glob:
                glob = glob % self.pipeline.options.subjid
            img = self.single_inimg(srcdir, glob, warn=True)
            if img is not None:
                LOG.info(f" - Saving additional map from {img.fpath} to {fname}")
                img.save(self.outfile(fname))

class DwiMoco(Module):
    def __init__(self, name="dwi_moco", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dwi_dir = self.kwargs.get("dwi_dir", "../fsort/dwi")
        dwi_glob = self.kwargs.get("dwi_glob", "dwi.nii.gz")

        dwi = self.single_inimg(dwi_dir, dwi_glob)
        if dwi is None:
            self.no_data(f"No DWI data found matching {dwi_dir}/{dwi_glob}")

        # Motion correct all bvals
        LOG.info(f" - Processing DWI data from {dwi.fname}")
        from ukat.mapping.diffusion import ADC
        adc_moco_mapper = ADC(dwi.data, dwi.affine, dwi.bval, ukrin_b=False, moco=True)
        moco_data = adc_moco_mapper.pixel_array_mean
        dwi.save_derived(moco_data, self.outfile("dwi_moco.nii.gz"), copy_bdata=False)
        bval_moco = np.unique(dwi.bval)
        np.savetxt(self.outfile('dwi_moco.bval'), np.expand_dims(bval_moco, 1).T, fmt='%.0f')

        moco_bvals = adc_moco_mapper.u_bvals
        np.savetxt(self.outfile('dwi_moco.bval'), np.expand_dims(moco_bvals, 1).T, fmt='%.0f')

class DwiAdc(Module):
    def __init__(self, name="dwi_adc", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        dwi_dir = self.kwargs.get("dwi_dir", "dwi_moco")
        dwi_glob = self.kwargs.get("dwi_glob", "dwi_moco.nii.gz")

        dwi = self.single_inimg(dwi_dir, dwi_glob, src=self.OUTPUT)
        if dwi is None:
            self.no_data(f"No DWI data found matching {dwi_dir}/{dwi_glob}")

        # Fit ADC using a limited number of bvals
        LOG.info(f" - Processing DWI data from {dwi.fname} (bvals: {dwi.bval})")
        from ukat.mapping.diffusion import ADC
        adc_mapper = ADC(dwi.data, dwi.affine, np.unique(dwi.bval), ukrin_b=True, moco=False)
        adc_mapper.to_nifti(self.outdir, base_file_name=dwi.fname_noext)


class AslMoco(Module):
    def __init__(self, name="asl_moco", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        asl_dir = self.kwargs.get("asl_dir", "../fsort/asl")
        asl_glob = self.kwargs.get("asl_glob", "asl*.nii.gz")

        asl_imgs = self.inimgs(asl_dir, asl_glob)
        if not asl_imgs:
            self.no_data(f"No ASL data found matching {asl_dir}/{asl_glob}")

        from ukat.mapping.perfusion import Perfusion
        for img in asl_imgs:
            LOG.info(f" - Processing ASL data from {img.fname}")
            perf_mapper = Perfusion(img.data, img.affine, moco=True)
            perf_mapper.to_nifti(self.outdir, base_file_name=img.fname_noext)
            moco_data = perf_mapper.pixel_array
            img.save_derived(moco_data, self.outfile(f'{img.fname_noext}_moco.nii.gz'))
