"""
FPROC: Modules for extracting statistics measures
"""

import logging
import os

import numpy as np
import radiomics
from dbdicom.wrappers.skimage import _volume_features
import nibabel as nib

from fproc.module import Module
from fproc import stats
from fproc.pipeline import ALL_MODULES

LOG = logging.getLogger(__name__)


class SegStats(Module):
    """
    A module which generates stats on parameters within segmentations
    """

    DEFAULT_LIMITS = {
        "3t": {
            "t1": {
                ("liver",): (500, 1300),
                ("spleen",): (900, 1660),
                ("pancreas",): (400, 1300),
                ("kidney", "cortex", "medulla", "tkv"): (1000, 2500),
            },
            "t2star": {
                ("liver",): (2, 70),
                ("spleen",): (2, 150),
                ("pancreas",): (2, 100),
                ("kidney", "cortex", "medulla", "tkv"): (2, 100),
            },
            "r2star": {
                ("liver",): (14, 500),
                ("spleen",): (6, 500),
                ("pancreas",): (10, 500),
                ("kidney", "cortex", "medulla", "tkv"): (10, 500),
            },
        },
        "1.5t": {
            "t1": {
                ("liver",): (400, 1350),
                ("spleen",): (800, 1350),
                ("pancreas",): (400, 1350),
                ("kidney", "cortex", "medulla", "tkv"): (900, 2400),
            },
            "t2star": {
                ("liver",): (2, 70),
                ("spleen",): (2, 150),
                ("pancreas",): (2, 100),
            },
            "r2star": {
                ("liver",): (14, 500),
                ("spleen",): (6, 500),
                ("pancreas",): (10, 500),
            },
        },
    }

    @staticmethod
    def _deps_from_dirs(segs, params):
        deps = []
        for spec in segs.values():
            if (
                isinstance(spec, dict)
                and isinstance(spec.get("dir", None), str)
                and spec["dir"]
            ):
                deps.append(spec["dir"])
        for spec in params.values():
            if (
                isinstance(spec, dict)
                and isinstance(spec.get("dir", None), str)
                and spec["dir"]
            ):
                deps.append(spec["dir"])
        return sorted(set(deps))

    def __init__(
        self,
        name="stats",
        segs=None,
        params=None,
        stats=None,
        out_name="stats.csv",
        default_limits=None,
        multi_mode="combine",
        allow_rotated=True,
        seg_volumes=False,
        overlays=True,
        **kwargs,
    ):
        segs = {} if segs is None else segs
        params = {} if params is None else params
        stats = [] if stats is None else stats
        deps = kwargs.pop("deps", None)
        if deps is None:
            deps = self._deps_from_dirs(segs, params)
        Module.__init__(self, name, deps=deps, **kwargs)
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
                if (params_segs is not None and seg not in params_segs) or (
                    seg_params is not None and param not in seg_params
                ):
                    LOG.debug(f" - Skipping segmentation {seg} for param {param}")
                    continue

                self._add_param_stats(
                    param, param_spec, seg, seg_spec, stat_names, values
                )

        stats_path = self.outfile(self.out_name)
        LOG.info(f" - Saving stats to {stats_path}")
        with open(stats_path, "w") as stats_file:
            for name, value in zip(stat_names, values):
                stats_file.write(f"{name},{str(value)}\n")

    def _add_param_stats(self, param, param_spec, seg, seg_spec, stat_names, values):
        stats_data, res_niis, n_found, best_count = [], [], 0, 0

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

        LOG.info(
            f" - Generating stats for param {param}, segmentation {seg} with limits {data_limits}"
        )
        LOG.debug(param_spec)
        LOG.debug(seg_spec)

        # Segmentation spec can be overridden, e.g. change glob or dir
        seg_overrides = param_spec.get("seg_overrides", {}).get(seg, {})
        seg_spec = seg_spec.copy()
        if seg_overrides:
            seg_spec.update(seg_overrides)
            LOG.debug(f" - Found segmentation overrides: {seg_overrides}: {seg_spec}")
        roi_id = seg_spec.get("roi_id", None)

        voxel_volume = 1.0  # In case there are no parameter images
        param_imgs = self._imgs(param_spec)
        if not param_imgs:
            LOG.warn(f" - No parameter images found for param {param}")
        for param_img in param_imgs:
            voxel_volume = param_img.voxel_volume
            for seg_img in self._imgs(seg_spec):
                seg_nii_res = self.resample(
                    seg_img, param_img, is_roi=True, allow_rotated=self.allow_rotated
                )
                res_data = seg_nii_res.get_fdata()
                if roi_id is None:
                    orig_mask = seg_img.data > 0
                    res_mask = res_data > 0
                else:
                    orig_mask = np.isclose(seg_img.data, roi_id)
                    res_mask = np.isclose(res_data, roi_id)

                orig_count = np.count_nonzero(orig_mask)
                res_count = np.count_nonzero(res_mask)
                n_found += 1 if res_count > 0 else 0
                LOG.debug(
                    f" - Param {param_img.fname}, Seg {seg_img.fname} count {res_count} orig {orig_count}"
                )
                if orig_count == 0:
                    LOG.warn(f"Segmentation {seg_img.fname} was empty")
                elif res_count == 0:
                    LOG.warn(
                        f"Segmentation {seg_img.fname} is empty after resampling to {param_img.fname}"
                    )
                if self.multi_mode == "best":
                    if res_count > best_count:
                        stats_data = [param_img.data[res_mask]]
                        res_niis = [seg_nii_res]
                        best_count = res_count
                elif self.multi_mode == "combine":
                    if res_count > 0:
                        stats_data.append(param_img.data[res_mask])
                        res_niis.append(seg_nii_res)
                if res_count > 0 and self.overlays:
                    self.lightbox(
                        param_img,
                        seg_img,
                        name=f"stats_{seg_img.fname_noext}_{param_img.fname_noext}_lightbox",
                    )

        if param_imgs:
            if n_found == 0:
                LOG.debug(" - No combination found with overlap")
            elif self.multi_mode == "best" and n_found != 1:
                LOG.warn(
                    f" - {n_found} combinations found with overlap - choosing best"
                )
            elif self.multi_mode == "combine":
                LOG.debug(
                    f" - Combining data from {n_found} overlapping parameter/segmentation maps"
                )

        for idx, res_img in enumerate(res_niis):
            res_path = self.outfile(f"{seg}_res_{param}_{idx+1}.nii.gz")
            LOG.debug(f" - Saving resampled segmentation to {res_path}")
            res_img.to_filename(res_path)

        if stats_data:
            stats_data = np.concatenate(stats_data)

        param_stats = stats.run(
            stats_data,
            stats=self.stats,
            data_limits=data_limits,
            voxel_volume=voxel_volume,
        )
        if "vol" in self.stats:
            param_stats["vol"] = param_stats["n"] * voxel_volume
        if "iqvol" in self.stats:
            param_stats["iqvol"] = param_stats["iqn"] * voxel_volume
        data_colname = param + "_" + seg
        for stat, value in param_stats.items():
            stat_names.append(stat + "_" + data_colname)
            values.append(value)

    def _add_seg_vols(self, seg, seg_spec, stat_names, values):
        LOG.info(f" - Adding N/volume for segmentation {seg}")
        n, vol = 0, 0
        roi_id = seg_spec.get("roi_id", None)
        for seg_img in self._imgs(seg_spec):
            if roi_id is None:
                nvox = np.count_nonzero(seg_img.data)
            else:
                nvox = np.count_nonzero(np.isclose(seg_img.data, roi_id))
            n += nvox
            vol += nvox * seg_img.voxel_volume
        stat_names.append(seg + "_n")
        values.append(n)
        stat_names.append(seg + "_vol")
        values.append(vol)

    def _imgs(self, spec):
        src, subdir, globexpr = (
            spec.get("src", self.pipeline.options.output),
            spec["dir"],
            spec["glob"],
        )
        imgs = self.inimgs(subdir, globexpr, src=src)
        if not imgs:
            LOG.warn(f" - No images found matching {globexpr} in {src}/{subdir}")
        return imgs


class AllRoiStats(SegStats):
    """
    Generate per-label segmentation volumes from labelled ROI images.
    """

    def __init__(self, name, roi_dir, roi_glob, roi_names=None, **kwargs):
        if "deps" in kwargs:
            raise TypeError("AllRoiStats does not accept 'deps' in kwargs")
        if "params" in kwargs:
            raise TypeError("AllRoiStats does not accept 'params' in kwargs")

        self.roi_src = kwargs.pop("roi_src", Module.OUTPUT)
        self.roi_names = (
            roi_names if roi_names is not None else kwargs.pop("roi_names", None)
        )
        SegStats.__init__(self, name, deps=[roi_dir], params={}, **kwargs)
        self.roi_dir = roi_dir
        self.roi_glob = roi_glob

    def process(self):
        roi_imgs = self.inimgs(self.roi_dir, self.roi_glob, src=self.roi_src)
        if not roi_imgs:
            self.no_data(f"No ROI images found matching {self.roi_glob} in {self.roi_dir}")
        self.segs = {}
        for roi_img in roi_imgs:
            vals = np.unique(roi_img.data)
            finite_vals = vals[np.isfinite(vals)]
            int_vals = finite_vals[np.isclose(finite_vals, np.round(finite_vals))]
            roi_ids = sorted({int(v) for v in int_vals if int(v) != 0})
            LOG.info(f" - Found ROI IDs in {roi_img.fname}: {roi_ids}")

            for roi_id in roi_ids:
                seg_name = f"{roi_img.fname_noext}_{roi_id}"
                if isinstance(self.roi_names, dict):
                    seg_name = self.roi_names.get(roi_id, seg_name)

                if seg_name in self.segs:
                    base_name = seg_name
                    seg_name = f"{base_name}_{roi_img.fname_noext}"
                    suffix = 2
                    while seg_name in self.segs:
                        seg_name = f"{base_name}_{roi_img.fname_noext}_{suffix}"
                        suffix += 1
                    LOG.warn(
                        f" - ROI name collision for '{base_name}' (id {roi_id}); using '{seg_name}'"
                    )

                self.segs[seg_name] = {
                    "dir": self.roi_dir,
                    "glob": roi_img.fname,
                    "src": self.roi_src,
                    "seg_volumes": True,
                    "roi_id": roi_id,
                }
        super().process()


class Radiomics(Module):
    def __init__(
        self,
        name="radiomics",
        segs=None,
        params=None,
        out_name="radiomics.csv",
        **kwargs,
    ):
        Module.__init__(self, name, **kwargs)
        segs = {} if segs is None else segs
        params = {} if params is None else params
        self.segs = segs
        self.params = params
        self.out_name = out_name

    def _get_imgs(self, spec, warn_none=True):
        src, subdir, globexpr = (
            spec.get("src", self.pipeline.options.output),
            spec["dir"],
            spec.get("fname", spec.get("glob", "")),
        )
        imgs = self.inimgs(subdir, globexpr, src=src)
        if not imgs and warn_none:
            LOG.warn(f" - No images found matching {spec}")
        return imgs

    def _get_img(self, spec, warn_none=True, warn_multiple=True):
        imgs = self._get_imgs(spec, warn_none=warn_none)
        if len(imgs) > 1 and warn_multiple:
            LOG.warn(f" - Multiple images found matching {spec} - returning first")
        return None if not imgs else imgs[0]

    def process(self):
        logger = logging.getLogger("radiomics")
        logger.handlers.clear()
        logger.propagate = False
        logger.addHandler(logging.FileHandler(self.outfile("radiomics.log")))

        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(
            geometryTolerance=1e-3
        )
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
            if isinstance(features, dict):
                extractor.enableFeaturesByName(**features)
            else:
                for feature in features:
                    extractor.enableFeatureClassByName(feature)
        else:
            extractor.enableAllFeatures()

        with open(self.outfile(self.out_name), "w") as f:
            for param_name, param_spec in self.params.items():
                LOG.info(f" - Extracting features for map: {param_name}")
                param_img = self._get_img(param_spec)
                if param_img is None:
                    LOG.warn("Map not found - skipping")
                    continue
                for seg_name, seg_spec in self.segs.items():
                    LOG.info(f" - Extracting features for segmentation: {seg_name}")
                    seg_imgs = self._get_imgs(seg_spec)
                    if not seg_imgs:
                        LOG.warn("No segmentation images found - skipping")
                        continue
                    LOG.info(
                        f" - Found {len(seg_imgs)} segmentation images: {','.join([s.fname for s in seg_imgs])}"
                    )
                    seg_img = seg_imgs[0]
                    seg_data = seg_img.data.astype(int)
                    for seg_img2 in seg_imgs[1:]:
                        seg_data = np.logical_or(seg_data, seg_img2.data.astype(int))
                    seg_data = seg_data.astype(int)

                    map_res = self.resample(
                        param_img, seg_img, is_roi=False, allow_rotated=True
                    )
                    if "vol" in param_spec:
                        data_vol = map_res.get_fdata()[..., param_spec["vol"]]
                        map_res = nib.Nifti1Image(
                            data_vol, map_res.affine, map_res.header
                        )
                    map_res_fpath = self.outfile(f"{param_name}_res_{seg_name}.nii.gz")
                    map_res.to_filename(map_res_fpath)
                    map_data = map_res.get_fdata()
                    while map_data.ndim > 3:
                        map_data = map_data.squeeze(-1)

                    seg_restricted = np.copy(seg_data)
                    minval, maxval = param_spec.get("minval", None), param_spec.get(
                        "maxval", None
                    )
                    if minval is not None:
                        seg_restricted[map_data < minval] = 0
                        LOG.info(f" - Restricting mask to value >= {minval}")
                    if maxval is not None:
                        seg_restricted[map_data > maxval] = 0
                        LOG.info(f" - Restricting mask to value <= {maxval}")
                    seg_restricted_fpath = self.outfile(
                        f"{seg_name}_restricted_{param_name}.nii.gz"
                    )
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
        stats_file = os.path.join(
            self.pipeline.options.output,
            self.kwargs.get("stats_file", "stats/stats.csv"),
        )
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
                    LOG.warn(
                        f"Error parsing stats file {stats_file} line {line}: {exc}"
                    )

        stats_path = self.outfile(self.kwargs.get("csv_file", "cmd.csv"))
        LOG.info(f" - Saving CMD stats to {stats_path}")
        with open(stats_path, "w") as stats_file:
            for key, values in cmd_dict.items():
                if medulla_id not in values or cortex_id not in values:
                    LOG.warn(
                        f"Failed to find both cortex and medulla data for key: {key}"
                    )
                    continue
                cmd = values[medulla_id] - values[cortex_id]
                stats_file.write(f"{key},{str(cmd)}\n")


class ShapeMetrics(Module):
    def __init__(self, name="shape_metrics", **kwargs):
        self._seg_dir = kwargs.get("seg_dir", None)
        Module.__init__(self, name, deps=[self._seg_dir], **kwargs)

    def process(self):
        METRICS_MAPPING = {
            "Surface area": "surf_area",
            "Surface area / volume": "surf_area_over_vol",
            "Volume": "vol",
            "Bounding box volume": "vol_bb",
            "Convex hull volume": "vol_ch",
            "Volume of holes": "vol_holes",
            "Extent": "extent",
            "Solidity": "solidity",
            "Compactness": "compactness",
            "Long axis length": "long_axis",
            "Short axis length": "short_axis",
            "Equivalent diameter": "equiv_diam",
            "Longest caliper diameter": "longest_diam",
            "Maximum depth": "max_depth",
            "Primary moment of inertia": "mi1",
            "Second moment of inertia": "mi2",
            "Third moment of inertia": "mi3",
            "Mean moment of inertia": "mi_mean",
            "Fractional anisotropy of inertia": "fa",
            "QC - Volume check": "volcheck",
        }

        if self._seg_dir is None:
            raise RuntimeError("Must provide seg_dir for shape metrics")

        segs = self.kwargs.get("segs", {})
        if not segs:
            self.no_data(f" - No segmentations provided to generate shape metrics")

        metrics = self.kwargs.get("metrics", [])

        csv_fname = self.kwargs.get("csv_fname", "shape_metrics.csv")
        LOG.info(f" - Saving shape metrics to {csv_fname}")
        with open(self.outfile(csv_fname), "w") as f:
            for name, glob in segs.items():
                img = self.single_inimg(
                    self._seg_dir, glob, src=self.kwargs.get("seg_src", self.OUTPUT)
                )
                if img is None:
                    LOG.warn(
                        f" - No segmentation found for {name} matching {self._seg_dir}/{glob}"
                    )
                    continue
                LOG.info(f" - Calculating shape metrics from {img.fname}")
                try:
                    vol_metrics = _volume_features(img.data, affine=img.affine)
                    for metric, value in vol_metrics.items():
                        if (
                            metrics
                            and metric not in metrics
                            and METRICS_MAPPING[metric] not in metrics
                        ):
                            continue
                        value, units = value
                        col_name = f"{name}_" + METRICS_MAPPING[metric]
                        f.write(f"{col_name},{value}\n")
                    if "surf_area_over_vol" in metrics:
                        f.write(
                            f"{name}_surf_area_over_vol,{vol_metrics['Surface area'][0] / vol_metrics['Volume'][0]}\n"
                        )
                except Exception as exc:
                    LOG.warn(f"Failed to calculate shape metrics: {exc}")


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


class KidneyCystStats(Module):
    def __init__(self, name="kidney_cyst_stats", **kwargs):
        self._cyst_dir = kwargs.get("cyst_dir", "seg_kidney_cyst_t2w")
        deps = kwargs.pop("deps", [self._cyst_dir])
        Module.__init__(self, name, deps=deps, **kwargs)

    def process(self):
        cyst_glob = self.kwargs.get("cyst_glob", "kidney_cyst_mask.nii.gz")
        cyst_src = self.kwargs.get("cyst_src", self.OUTPUT)
        cyst_seg = self.single_inimg(self._cyst_dir, cyst_glob, src=cyst_src)
        if cyst_seg is None:
            self.no_data("No T2w kidney cyst segmentation found to clean")

        # Count number of cysts and volume
        cyst_blobs = [
            b for b in self.blobs_by_size(cyst_seg.data) if np.count_nonzero(b) > 1
        ]
        num_cysts = len(cyst_blobs)
        blob_sizes = [np.count_nonzero(b) for b in cyst_blobs]
        if num_cysts == 0:
            LOG.info(" - No cysts found with size > 1 voxel")
            total_volume, vol_mean, vol_min, vol_max = 0, 0, 0, 0
        else:
            total_volume = sum(blob_sizes) * cyst_seg.voxel_volume
            vol_mean = total_volume / num_cysts
            vol_min, vol_max = (
                min(blob_sizes) * cyst_seg.voxel_volume,
                max(blob_sizes) * cyst_seg.voxel_volume,
            )

        suffix = self.kwargs.get("suffix", "")
        if suffix:
            prefix = f"kidney_cyst_{suffix}"
        else:
            prefix = f"kidney_cyst"

        vol_cats = self.kwargs.get(
            "vol_cats",
            [
                300,
                200,
                100,
                50,
                40,
                30,
                20,
                10,
                4.8,
                3.6,
                2.4,
                1.2,
                0.56,
                0.45,
                0.34,
                0.23,
                0.12,
                0.045,
                0,
            ],
        )
        gt_vol_cats = self.kwargs.get("gt_vol_cats", [0.23, 0.12, 0.045])
        with open(self.outfile("kidney_cyst.csv"), "w") as f:
            f.write(f"{prefix}_vol,{total_volume}\n")
            f.write(f"{prefix}_n,{num_cysts}\n")
            f.write(f"{prefix}_vol_mean,{vol_mean}\n")
            f.write(f"{prefix}_vol_min,{vol_min}\n")
            f.write(f"{prefix}_vol_max,{vol_max}\n")

            cyst_cats = []
            for blob_size in blob_sizes:
                for cat_id, min_vol in enumerate(vol_cats):
                    if blob_size * cyst_seg.voxel_volume > min_vol:
                        cyst_cats.append(cat_id)
                        break

            prev_min_vol = None
            for cat_id, min_vol in enumerate(vol_cats):
                num_cat = cyst_cats.count(cat_id)
                if prev_min_vol:
                    f.write(f"{prefix}_vol_gt_{min_vol}_lt_{prev_min_vol},{num_cat}\n")
                else:
                    f.write(f"{prefix}_vol_gt_{min_vol},{num_cat}\n")
                prev_min_vol = min_vol

            for min_vol in gt_vol_cats:
                total_vol, total_num = 0, 0
                for blob_size in blob_sizes:
                    cyst_vol = blob_size * cyst_seg.voxel_volume
                    if cyst_vol > min_vol:
                        total_vol += cyst_vol
                        total_num += 1
                f.write(f"{prefix}_n_gt_{min_vol},{total_num}\n")
                f.write(f"{prefix}_vol_gt_{min_vol},{total_vol}\n")

        self.runcmd(
            [
                "cluster",
                f"--in={cyst_seg.fpath}",
                f"--thresh=1",
                f"--oindex={self.outfile('clusterindex')}",
                f"--minextent=2",
            ],
            logfile="cluster.log",
        )


class SegVolumeDiffs(Module):
    def __init__(self, name="seg_volume_diffs", **kwargs):
        Module.__init__(self, name, deps=ALL_MODULES, **kwargs)

    def process(self):
        stats_files = self.kwargs.get("stats_files", "stats/stats.csv")
        if not isinstance(stats_files, list):
            stats_files = [stats_files]
        volumes = {}
        for stats_file in stats_files:
            stats_file = os.path.join(self.pipeline.options.output, stats_file)
            if not os.path.isfile(stats_file):
                LOG.warn(f" - Not a file: {stats_file}")
                continue
            with open(stats_file) as f:
                for line in f:
                    try:
                        key, value = line.split(",")
                        value = float(value)
                        if key in volumes:
                            LOG.warn(
                                f" - Duplicate volume entry for {key} in {stats_file} - ignoring"
                            )
                            continue
                        volumes[key] = value
                    except:
                        LOG.warn(f"Error parsing stats file {stats_file} line {line}")

        out_path = self.outfile(self.kwargs.get("csv_file", "diffs.csv"))
        diffs = self.kwargs.get("diffs", {})
        with open(out_path, "w") as f:
            for output_name, input_segs in diffs.items():
                first, second = input_segs
                if first not in volumes or second not in volumes:
                    LOG.warn(
                        f" - Could not find volumes for diff {output_name}: {first}, {second}"
                    )
                    continue
                diff = volumes[first] - volumes[second]
                f.write(f"{output_name},{diff}\n")
