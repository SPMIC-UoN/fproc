"""
FPROC: Modules for extracting statistics measures
"""
import logging
import os

import numpy as np
import radiomics
from dbdicom.wrappers.skimage import _volume_features

from fproc.module import Module
from fproc import stats

LOG = logging.getLogger(__name__)

class SegStats(Module):
    """
    A module which generates stats on parameters within segmentations
    """

    DEFAULT_LIMITS = {
        "3t" : {
            "t1" : {
                ("liver",) : (500, 1300),
                ("spleen",) : (900, 1660),
                ("pancreas",) : (400, 1300),
                ("kidney", "cortex", "medulla", "tkv") : (1000, 2500),
            },
            "t2star" : {
                ("liver",) : (2, 70),
                ("spleen",) : (2, 150),
                ("pancreas",) : (2, 100),
                ("kidney", "cortex", "medulla", "tkv") : (2, 100),
            },
            "r2star" : {
                ("liver",) : (14, 500),
                ("spleen",) : (6, 500),
                ("pancreas",) : (10, 500),
                ("kidney", "cortex", "medulla", "tkv") : (10, 500),
            },
        },
        "1.5t" : {
            "t1" : {
                ("liver",) : (400, 1350),
                ("spleen",) : (800, 1350),
                ("pancreas",) : (400, 1350),
                ("kidney", "cortex", "medulla", "tkv") : (900, 2400),
            },
            "t2star" : {
                ("liver",) : (2, 70),
                ("spleen",) : (2, 150),
                ("pancreas",) : (2, 100),
            },
            "r2star" : {
                ("liver",) : (14, 500),
                ("spleen",) : (6, 500),
                ("pancreas",) : (10, 500),
            },
        },
    }

    def __init__(self, name="stats", segs={}, params={}, stats=[], out_name="stats.csv", default_limits=None, multi_mode="combine", allow_rotated=True, seg_volumes=False, overlays=True):
        Module.__init__(self, name)
        self.segs = segs
        self.params = params
        self.stats = stats
        self.out_name = out_name
        self.default_limits = default_limits
        self.multi_mode = multi_mode
        self.allow_rotated = allow_rotated
        self.seg_volumes = seg_volumes
        self.overlays = overlays
        if self.multi_mode not in ("best", "combine"):
            raise RuntimeError(f"Multi mode not recognized: {self.multi_mode}")

    def process(self):
        stat_names, values = [], []
        for seg, seg_spec in self.segs.items():
            if seg_spec.get("seg_volumes", self.seg_volumes):
                self._add_seg_vols(seg, seg_spec, stat_names, values)

        for param, param_spec in self.params.items():
            params_segs = param_spec.get("segs", None)
            for seg, seg_spec in self.segs.items():
                seg_params = seg_spec.get("params", None)
                if (params_segs is not None and seg not in params_segs) or (seg_params is not None and param not in seg_params):
                    LOG.debug(f" - Skipping segmentation {seg} for param {param}")
                    continue

                self._add_param_stats(param, param_spec, seg, seg_spec, stat_names, values)

        stats_path = self.outfile(self.out_name)
        LOG.info(f" - Saving stats to {stats_path}")
        with open(stats_path, "w") as stats_file:
            for name, value in zip(stat_names, values):
                stats_file.write(f"{name},{str(value)}\n")

    def _add_param_stats(self, param, param_spec, seg, seg_spec, stat_names, values):
        stats_data, res_niis, n_found, best_count = [], [], 0, 0

        # Segmentation spec can be overridden, e.g. change glob or dir
        seg_overrides = param_spec.get("seg_overrides", {}).get(seg, {})
        seg_spec = seg_spec.copy()
        if seg_overrides:
            seg_spec.update(seg_overrides)
            LOG.debug(f" - Found segmentation overrides: {seg_overrides}: {seg_spec}")

        LOG.debug(f" - Adding stats for param {param}, segmentation {seg}")
        LOG.debug(param_spec)
        LOG.debug(seg_spec)

        voxel_volume = 1.0  # In case there are no parameter images
        param_imgs = self._imgs(param_spec)
        for param_img in param_imgs:
            voxel_volume = param_img.voxel_volume
            for seg_img in self._imgs(seg_spec):
                seg_nii_res = self.resample(seg_img, param_img, is_roi=True, allow_rotated=self.allow_rotated)
                res_data = seg_nii_res.get_fdata()
                orig_count = np.count_nonzero(seg_img.data)
                res_count = np.count_nonzero(res_data)
                n_found += 1 if res_count > 0 else 0
                LOG.debug(f" - Param {param_img.fname}, Seg {seg_img.fname} count {res_count} orig {orig_count}")
                if self.multi_mode == "best":
                    if res_count > best_count:
                        stats_data = [param_img.data[res_data > 0]]
                        res_niis = [seg_nii_res]
                elif self.multi_mode == "combine":
                    if res_count > 0:
                        stats_data.append(param_img.data[res_data > 0])
                        res_niis.append(seg_nii_res)
                if res_count > 0 and self.overlays:
                    self.lightbox(param_img, seg_img, name=f"stats_{seg_img.fname_noext}_{param_img.fname_noext}_lightbox")

        if param_imgs:
            if n_found == 0:
                LOG.warn(" - No combination found with overlap")
            elif self.multi_mode == "best" and n_found != 1:
                LOG.warn(f" - {n_found} combinations found with overlap - choosing best")
            elif self.multi_mode == "combine":
                LOG.debug(f" - Combining data from {n_found} overlapping parameter/segmentation maps")

        for idx, res_img in enumerate(res_niis):
            res_path = self.outfile(f"{seg}_res_{param}_{idx+1}.nii.gz")
            LOG.debug(f" - Saving resampled segmentation to {res_path}")
            res_img.to_filename(res_path)

        if stats_data:
            stats_data = np.concatenate(stats_data)

        # Data limits can be set as generic limits or specific to particular seg
        # There are also defaults for specific organs/params keyed by field strength
        # specified using limits="default_3t" for example
        data_limits = param_spec.get("limits", (None, None))
        if isinstance(data_limits, dict):
            data_limits = data_limits.get(seg, data_limits.get("", (None, None)))

        if data_limits == (None, None) and self.default_limits:
            default_limits = self.DEFAULT_LIMITS.get(self.default_limits, {})
            for param_substr, param_limits in default_limits.items():
                if param_substr in param:
                    for seg_substrs, seg_limits in param_limits.items():
                        for seg_substr in seg_substrs:
                            if seg_substr in seg:
                                data_limits = seg_limits
                                LOG.debug(f" - Using default limits: {data_limits}")
                                break

        if len(data_limits) != 2:
            raise RuntimeError(f"Invalid data limits: {data_limits}")

        LOG.info(f" - Generating stats for param {param}, segmentation {seg} with limits {data_limits}")
        param_stats = stats.run(stats_data, stats=self.stats, data_limits=data_limits, voxel_volume=voxel_volume)
        if "vol" in self.stats:
            param_stats["vol"] = param_stats["n"] * voxel_volume
        if "iqvol" in self.stats:
            param_stats["iqvol"] = param_stats["iqn"] * voxel_volume
        data_colname = param + "_" + seg
        for stat, value in param_stats.items():
            stat_names.append(stat + "_"+ data_colname)
            values.append(value)

    def _add_seg_vols(self, seg, seg_spec, stat_names, values):
        LOG.info(f" - Adding N/volume for segmentation {seg}")
        n, vol = 0, 0
        for seg_img in self._imgs(seg_spec):
            nvox = np.count_nonzero(seg_img.data)
            n += nvox
            vol += nvox * seg_img.voxel_volume
        stat_names.append(seg + "_n")
        values.append(n)
        stat_names.append(seg + "_vol")
        values.append(vol)

    def _imgs(self, spec):
        src, subdir, globexpr = spec.get("src", self.pipeline.options.output), spec["dir"], spec["glob"]
        imgs = self.inimgs(subdir, globexpr, src=src)
        if not imgs:
            LOG.warn(f" - No images found matching {globexpr} in {src}/{subdir}")
        return imgs

class Radiomics(Module):
    def __init__(self, name="radiomics", segs={}, params={}, out_name="radiomics.csv", **kwargs):
        Module.__init__(self, name, **kwargs)
        self.segs = segs
        self.params = params
        self.out_name = out_name

    def _get_img(self, spec, warn_none=True, warn_multiple=True):
        src, subdir, globexpr = spec.get("src", self.pipeline.options.output), spec["dir"], spec.get("fname", spec.get("glob", ""))
        imgs = self.inimgs(subdir, globexpr, src=src)
        if not imgs:
            if warn_none:
                LOG.warn(f" - No images found matching {globexpr} in {src}/{subdir}")
            return None
        elif len(imgs) > 1 and warn_multiple:
            LOG.warn(f" - Multiple images found matching {globexpr} in {src}/{subdir} - returning first")
        return imgs[0]

    def process(self):
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(geometryTolerance=1e-3)
        image_types = self.kwargs.get("image_types", None)
        if image_types:
            extractor.disableAllImageTypes()
            for image_type in image_types:
                extractor.enableImageTypeByName(image_type)
        else:
            extractor.enableAllImageTypes()
        features = self.kwargs.get("features", None)
        if features:
            extractor.disableAllFeatures()
            for feature in features:
                extractor.enableFeatureClassByName(feature)
        else:
            extractor.enableAllFeatures()

        with open(self.outfile(self.out_name), "w") as f:
            for param_name, param_spec in self.params.items():
                LOG.info(f" - Extracting features for map: {param_name}")
                param_img = self._get_img(param_spec)
                if param_img is None:
                    continue
                for seg_name, seg_spec in self.segs.items():
                    LOG.info(f" - Extracting features for segmentation: {seg_name}")
                    seg_img = self._get_img(seg_spec)
                    if seg_img is None:
                        continue
                    
                    map_res = self.resample(param_img, seg_img, is_roi=False, allow_rotated=True)
                    map_res_fpath = self.outfile(f"{param_name}_res_{seg_name}.nii.gz")
                    map_res.to_filename(map_res_fpath)
                    map_data = map_res.get_fdata().squeeze()
                    seg_restricted = np.copy(seg_img.data)
                    minval, maxval = param_spec.get("minval", None), param_spec.get("maxval", None)
                    if minval is not None:
                        seg_restricted[map_data < minval] = 0
                    if maxval is not None:
                        seg_restricted[map_data > maxval] = 0
                    seg_restricted_fpath = self.outfile(f"{seg_name}_restricted_{param_name}.nii.gz")
                    seg_img.save_derived(seg_restricted, seg_restricted_fpath)

                    try:
                        results = extractor.execute(map_res_fpath, seg_restricted_fpath)
                        for k, v in results.items():
                            if k.startswith("diagnostics"):
                                continue
                            f.write(f"{param_name}_{seg_name}_{k},{v}\n")
                    except Exception as exc:
                        LOG.warn(f" - Failed to extract features: {exc}")


class CMD(Module):
    def __init__(self, name="cmd", **kwargs):
        Module.__init__(self, name, **kwargs)
        
    def process(self):
        stats_file = os.path.join(self.pipeline.options.output, self.kwargs.get("stats_file", "stats/stats.csv"))
        cortex_id = self.kwargs.get("cortex_id", "kidney_cortex")
        medulla_id = self.kwargs.get("medulla_id", "kidney_medulla")

        params = self.kwargs.get("cmd_params", None)
        skip_params = self.kwargs.get("skip_params", None)
        cmd_dict = {}
        with open(stats_file) as f:
            for line in f:
                try:
                    key, value = line.split(",")
                    value = float(value)
                    if params:
                        found = [param in key for param in params]
                        if not any(found):
                            LOG.debug(f" - Ignoring {key} - not in {params}")
                            continue
                    if skip_params:
                        found = [param in key for param in skip_params]
                        if any(found):
                            LOG.debug(f" - Ignoring {key} - in {skip_params}")
                            continue
                    if "iqmean" not in key and "median" not in key:
                        # Only calculate CMD for IQ mean and median
                        continue
                    for struc in (cortex_id, medulla_id):
                        if struc in key:
                            generic_key = key.replace(struc, "cmd")
                            if generic_key not in cmd_dict:
                                cmd_dict[generic_key] = {}
                            cmd_dict[generic_key][struc] = value
                except Exception as exc:
                    LOG.warn(f"Error parsing stats file {stats_file} line {line}: {exc}")

        stats_path = self.outfile(self.kwargs.get("csv_file", "cmd.csv"))
        LOG.info(f" - Saving CMD stats to {stats_path}")
        with open(stats_path, "w") as stats_file:
            for key, values in cmd_dict.items():
                if medulla_id not in values or cortex_id not in values:
                    LOG.warn(f"Failed to find both cortex and medulla data for key: {key}")
                    continue
                cmd = values[medulla_id] - values[cortex_id]
                stats_file.write(f"{key},{str(cmd)}\n")


class ShapeMetrics(Module):
    def __init__(self, name="shape_metrics", **kwargs):
        Module.__init__(self, name, **kwargs)

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

        seg_dir = self.kwargs.get("seg_dir", None)
        if seg_dir is None:
            raise RuntimeError("Must provide seg_dir for shape metrics")

        seg_globs = self.kwargs.get("seg_globs", "seg*.nii.gz")
        if not seg_globs:
            seg_globs = [self.kwargs.get("seg_glob", "seg*.nii.gz")]

        all_segs = []
        for seg_glob in seg_globs:
            all_segs.extend(self.inimgs(seg_dir, seg_glob, src=self.OUTPUT))

        if not all_segs:
            self.no_data(f" - No segmentations found in {seg_dir} matching {seg_globs}")

        csv_fname = self.kwargs.get("csv_fname", f"{seg_dir}_shape_metrics.csv")
        LOG.info(f" - Saving shape metrics to {csv_fname}")
        with open(self.outfile(csv_fname), "w") as f:
            for seg_img in all_segs:
                LOG.info(f" - Calculating T2w shape metrics from {seg_img.fname}")
                try:
                    vol_metrics = _volume_features(seg_img.data, affine=seg_img.affine)
                except Exception as exc:
                    LOG.warn(f"Failed to calculate shape metrics: {exc}")
                for metric, value in vol_metrics.items():
                    value, units = value
                    col_name = f"{seg_img.fname_noext}_" + METRICS_MAPPING[metric]
                    f.write(f"{col_name},{value}\n")

class ISNR(Module):
    def __init__(self, name="isnr", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        src = self.kwargs.get("src", self.OUTPUT)
        fname = self.kwargs.get("fname", "isnr.csv")
        img_globs = self.kwargs.get("imgs", {})
        if not img_globs:
            self.no_data("No images specified for ISNR calculation")
        
        with open(self.outfile(fname), "w") as f:
            for dir, glob in img_globs.items():
                imgs = self.inimgs(dir, glob, src=src)
                if not imgs:
                    LOG.warn(f" - No images found matching {glob} in {src}/{dir}")
                    continue
                for img in imgs:
                    from ukat.qa import snr
                    isnr = snr.Isnr(img.data, img.affine).isnr
                    f.write(f"{img.fname_noext}_isnr,{isnr}\n")
