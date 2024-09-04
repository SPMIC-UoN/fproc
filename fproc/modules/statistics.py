"""
FPROC: Modules for extracting statistics measures
"""
import logging

import numpy as np
import radiomics

from fproc.module import Module
from fproc import stats

LOG = logging.getLogger(__name__)

class SegStats(Module):
    """
    A module which generates stats on parameters within segmentations
    """
    def __init__(self, name="stats", segs={}, params={}, stats=[], out_name="stats.csv", multi_mode="combine", allow_rotated=True, seg_volumes=False, overlays=True):
        Module.__init__(self, name)
        self.segs = segs
        self.params = params
        self.stats = stats
        self.out_name = out_name
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

        LOG.info(f" - Generating stats for param {param}, segmentation {seg}")
        LOG.debug(param_spec)
        LOG.debug(seg_spec)

        voxel_volume = 1.0  # In case there are no parameter images
        for param_img in self._imgs(param_spec):
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
                    self.lightbox(param_img, seg_img, name=f"{seg_img.fname_noext}_{param_img.fname_noext}_lightbox")

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

        param_stats = stats.run(stats_data, stats=self.stats, data_limits=param_spec.get("limits", (None, None)), voxel_volume=voxel_volume)
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
    def __init__(self, name="radiomics", segs={}, params={}, out_name="radiomics.csv"):
        Module.__init__(self, name)
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
        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllImageTypes()

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
                        seg_restricted[map_data < minval] = minval
                    if maxval is not None:
                        seg_restricted[map_data > maxval] = maxval
                    seg_restricted_fpath = self.outfile(f"{seg_name}_restricted.nii.gz")
                    seg_img.save_derived(seg_restricted, seg_restricted_fpath)

                    results = extractor.execute(map_res_fpath, seg_restricted_fpath)
                    for k, v in results.items():
                        if k.startswith("diagnostics"):
                            continue
                        f.write(f"{param_name}_{seg_name}_{k},{v}\n")
