"""
FPROC: Base class for processing module
"""
import glob
import logging
import math
import os
import shutil
import subprocess

from fsort.image_file import ImageFile

import numpy as np
import scipy
import nibabel as nib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from . import stats

LOG = logging.getLogger(__name__)

class ModuleError(RuntimeError):
    pass

class Module:
    """
    A processing module
    """

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def __init__(self, name):
        self.name = name
        self.pipeline = None

    def run(self, pipeline):
        self.pipeline = pipeline
        self.outdir = os.path.abspath(os.path.normpath(os.path.join(self.pipeline.options.output, self.name)))
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)
        os.makedirs(self.outdir)
        self.process()

    def process(self):
        raise NotImplementedError()

    def indir(self, name):
        if self.pipeline is None:
            raise RuntimeError("No pipeline context")
        return os.path.abspath(os.path.normpath(os.path.join(self.pipeline.options.input, name)))

    def infile(self, dir, name, check=True, warn=False, src=None, is_depfile=False):
        if not src and is_depfile:
            src = self.pipeline.options.output
        elif not src:
            src = self.pipeline.options.input
        elif src == self.INPUT:
            src = self.pipeline.options.input
        elif src == self.OUTPUT:
            src = self.pipeline.options.output

        LOG.debug(f"infile: {src} {dir} {name}")
        indir = os.path.abspath(os.path.normpath(os.path.join(src, dir)))
        fpath = os.path.join(indir, name)
        if check and not os.path.exists(fpath):
            if warn:
                LOG.warn(f" - Expected file {fpath} did not exist")
            else:
                raise ModuleError(f" - Expected file {fpath} did not exist")
        else:
            LOG.debug(f"infile: {fpath}")
            return fpath
        
    def infiles(self, dir, globexpr, src=None, is_depfile=False):
        return sorted(glob.glob(self.infile(dir, globexpr, check=False, src=src, is_depfile=is_depfile)))

    def inimg(self, dir, name, check=True, warn=False, src=None, is_depfile=False):
        fpath = self.infile(dir, name, check=check, warn=warn, src=src, is_depfile=is_depfile)
        if not os.path.exists(fpath) and not check:
            return None
        else:
            return ImageFile(fpath, warn_json=False)

    def inimgs(self, dir, globexpr, src=None, is_depfile=False):
        return [ImageFile(f, warn_json=False) for f in self.infiles(dir, globexpr, src=src, is_depfile=is_depfile)]

    def outfile(self, name):
        return os.path.join(self.outdir, name)

    def copyinput(self, dir, glob):
        imgs = self.inimgs(dir, glob)
        ret = []
        for img in imgs:
            ret.append(img.save_derived(img.data, self.outfile(img.fname)))
        return ret

    def runcmd(self, cmd, logfile):
        LOG.info(cmd)
        with open(os.path.join(self.outdir, logfile), "w") as f:
            retval = subprocess.call(cmd, stdout=f, stderr=f)
        if retval != 0:
            LOG.warn(f" - Command {cmd} returned non-zero exit state {retval}")
        return retval

    def bad_data(self, reason):
        raise ModuleError(f"Bad data: {reason}")
    
    def no_data(self, reason):
        raise ModuleError(f"Can't generate output - no input data: {reason}")

    def resample(self, src, tgt, is_roi, allow_rotated=False):
        """
        Resample an image onto the grid from a target image
        """
        data_src = src.nii.get_fdata()
        data_tgt = tgt.nii.get_fdata()
        while data_tgt.ndim < 3:
            data_tgt = data_tgt[..., np.newaxis]
        while data_src.ndim < 3:
            data_src = data_src[..., np.newaxis]
        
        tmatrix = np.dot(np.linalg.inv(tgt.affine), src.affine)
        tmatrix = np.linalg.inv(tmatrix)
        affine = tmatrix[:3, :3]
        offset = list(tmatrix[:3, 3])
        output_shape = list(data_tgt.shape[:3])

        if data_src.ndim == 4:
            # Make 4D affine with identity transform in 4th dimension
            affine = np.append(affine, [[0, 0, 0]], 0)
            affine = np.append(affine, [[0], [0], [0], [1]], 1)
            offset.append(0)
            output_shape.append(data_src.shape[3])

        LOG.debug(f"Resampling from\n{src.affine}\n{data_src.shape}")
        LOG.debug(f"To\n{tgt.affine}\n{data_tgt.shape}")
        LOG.debug(f"Net\n{affine}\n{offset}")
        same = np.allclose(src.affine, tgt.affine) and list(data_src.shape[:3]) == output_shape[:3]
        LOG.debug(f"Same: {same}")
        if same:
            return src.nii
        
        if self.is_diagonal(affine):
            # Use faster sequence mode
            affine = np.diagonal(affine)
            LOG.debug(affine)
            LOG.debug(output_shape)
            LOG.debug(np.min(data_src))
            LOG.debug(np.max(data_src))
            res_data = scipy.ndimage.affine_transform(data_src, affine, offset=offset,
                                                    output_shape=output_shape, order=0 if is_roi else 1, mode='grid-constant')
        elif not allow_rotated:
            LOG.warn(f"Data is rotated relative to segmentation - will not use this segmentation")
            res_data = np.zeros(output_shape)
        else:
            LOG.debug(affine)
            LOG.debug(output_shape)
            LOG.debug(np.min(data_src), np.max(data_src))
            res_data = scipy.ndimage.affine_transform(data_src, affine, offset=offset,
                                                    output_shape=output_shape, order=0 if is_roi else 1, mode='grid-constant')

        if is_roi:
            res_data = res_data.astype(np.uint8)
        return nib.Nifti1Image(res_data, tgt.affine, tgt.nii.header)

    def is_diagonal(self, mat):
        """
        :return: True if mat is diagonal, to within a tolerance of ``EQ_TOL``
        """
        EQ_TOL = 1e-3
        return np.all(np.abs(mat - np.diag(np.diag(mat))) < EQ_TOL)

    def lightbox(self, img, mask, name, tight=False):
        """
        Generate a lightbox overlay of a mask onto an image
        """
        if isinstance(img, ImageFile) and isinstance(mask, ImageFile):
            img = self.resample(img, mask, is_roi=False, allow_rotated=True).get_fdata()
        if isinstance(img, ImageFile):
            img = img.data
        if isinstance(mask, ImageFile):
            mask = mask.data

        while mask.ndim < 3:
            mask = mask[..., np.newaxis]
        while img.ndim < 3:
            img = img[..., np.newaxis]
        while mask.ndim > 3:
            if mask.shape[-1] == 1:
                mask = np.squeeze(mask, -1)
            else:
                raise ValueError("Mask is not 3D")
        while img.ndim > 3:
            if img.shape[-1] == 1:
                img = np.squeeze(img, -1)
            else:
                raise ValueError("Image is not 3D")

        if mask.shape != img.shape:
            LOG.warn(f"Mask and image different shapes: {mask.shape}, {img.shape} - will not generate lightbox image {name}")
            return

        # If single slice make sure this is the last dimension
        single_slice_dims = [d for d in range(3) if mask.shape[d] == 1]
        if single_slice_dims:
            if single_slice_dims[0] == 0:
                mask = np.transpose(mask, (1, 2, 0))
                img = np.transpose(img, (1, 2, 0))
            elif single_slice_dims[0] == 1:
                mask = np.transpose(mask, (0, 2, 1))
                img = np.transpose(img, (0, 2, 1))

        bbox = [slice(None)] * 3
        min_slice, max_slice = 0, mask.shape[2]
        if np.any(mask):
            for dim in range(3):
                axes = [i for i in range(3) if i != dim]
                nonzero = np.any(mask, axis=tuple(axes))
                bb_start, bb_end = np.where(nonzero)[0][[0, -1]]
                # Pad bounding box but not for slices
                if dim != 2:
                    bb_start, bb_end = max(0, bb_start-5), min(mask.shape[dim]-1, bb_end+5)
                bbox[dim] = slice(bb_start, bb_end+1)

            min_slice, max_slice = bbox[2].start, bbox[2].stop

        num_slices = max_slice - min_slice
        grid_size = int(math.ceil(math.sqrt(num_slices)))

        fig = Figure(figsize=(5, 5), dpi=200)
        FigureCanvas(fig)
        for slice_idx in range(num_slices):
            axes = fig.add_subplot(grid_size, grid_size, slice_idx+1)
            axes.set_yticklabels([])
            axes.set_xticklabels([])
            axes.set_xticks([])
            axes.set_yticks([])

            if tight:
                data_slice = img[bbox[0], bbox[1], slice_idx + min_slice].T
                mask_slice = mask[bbox[0], bbox[1], slice_idx + min_slice].T
            else:
                data_slice = img[:, :, slice_idx + min_slice].T
                mask_slice = mask[:, :, slice_idx + min_slice].T
            axes.imshow(data_slice, cmap='gray')

            mask_slice = np.ma.masked_array(mask_slice, mask_slice == 0)
            axes.imshow(mask_slice, cmap='Reds', vmin=0, vmax=1)

            # Reverse y-axis so anterior is as the top
            axes.set_ylim(axes.get_ylim()[::-1])
            
            if self.pipeline.options.subjid and slice_idx == num_slices-1:
                axes.text(0, 0, self.pipeline.options.subjid, bbox={"fill" : True, "edgecolor": 'black', "linewidth" : 1, "facecolor" : "white"})

        fig.subplots_adjust(wspace=0, hspace=0.05)
        fig.savefig(self.outfile(name + ".png"), bbox_inches='tight')

    def matching_img(self, img, candidates):
        matches = []
        for candidate in candidates:
            if candidate.affine_matches(img):
                matches.append(candidate)
        if not matches:
            LOG.warn(f" - Could not find matching image for {img.fname} in {candidates}")
            return None
        elif len(matches) > 1:
            LOG.warn(f" - Multiple matching images for {img.fname} in {candidates} - returning first which is {matches[0].fname}")
        return matches[0]

class CopyModule(Module):
    """
    A module which just copies an input file
    """
    def __init__(self, name, in_dir=None, in_name=None, out_name=None):
        Module.__init__(self, name)
        self.in_dir = in_dir if in_dir is not None else name
        self.in_name = in_name if in_name is not None else name
        self.out_name = out_name if out_name is not None else name

    def process(self):
        self.img = self.inimg(self.in_dir, f"{self.in_name}.nii.gz")
        self.img.save_derived(self.img.nii.get_fdata(), self.outfile(f"{self.out_name}.nii.gz"))

class StatsModule(Module):
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
