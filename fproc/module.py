"""
FPROC: Base class for processing module
"""

import glob
import logging
import math
import os
import shutil
import subprocess

import nibabel as nib
import numpy as np
import scipy
import skimage
from fsort.image_file import ImageFile
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

    def __init__(self, name, **kwargs):
        self.name = name
        self.pipeline = None
        self.kwargs = kwargs
        self.deps = self.kwargs.pop("deps", [])

    def run(self, pipeline):
        self.pipeline = pipeline
        self.outdir = os.path.abspath(
            os.path.normpath(os.path.join(self.pipeline.options.output, self.name))
        )
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)
        os.makedirs(self.outdir)
        self.process()

    def process(self):
        raise NotImplementedError()

    def indir(self, name):
        if self.pipeline is None:
            raise RuntimeError("No pipeline context")
        return os.path.abspath(
            os.path.normpath(os.path.join(self.pipeline.options.input, name))
        )

    def last_done(self, pipeline_outdir):
        outdir = os.path.join(pipeline_outdir, self.name)
        donefile = os.path.join(outdir, "done.txt")
        if os.path.isfile(donefile):
            try:
                return os.path.getmtime(donefile)
            except Exception as e:
                LOG.warning(f"Error getting last done time from {donefile}: {e}")
                return None
        else:
            return None

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
        return sorted(
            glob.glob(
                self.infile(dir, globexpr, check=False, src=src, is_depfile=is_depfile)
            )
        )

    def inimg(self, dir, name, check=True, warn=False, src=None, is_depfile=False):
        fpath = self.infile(
            dir, name, check=check, warn=warn, src=src, is_depfile=is_depfile
        )
        if not os.path.exists(fpath) and not check:
            return None
        else:
            return ImageFile(fpath, warn_json=False)

    def single_inimg(self, dir, globexpr, src=None, is_depfile=False, warn=True):
        imgs = self.inimgs(dir, globexpr, src, is_depfile)
        if not imgs:
            if warn:
                LOG.warn(f" - No images found matching {dir}/{globexpr} in {src}")
            return None
        elif warn and len(imgs) > 1:
            LOG.warn(
                f" - Multiple reference images found matching {dir}/{globexpr} in {src}"
            )
            LOG.warn(f" - using first: {imgs[0].fname}")
        return imgs[0]

    def inimgs(self, dir, globexpr, src=None, is_depfile=False):
        return [
            ImageFile(f, warn_json=False)
            for f in self.infiles(dir, globexpr, src=src, is_depfile=is_depfile)
        ]

    def outfile(self, name):
        return os.path.join(self.outdir, name)

    def copyinput(self, dir, glob):
        imgs = self.inimgs(dir, glob)
        ret = []
        for img in imgs:
            ret.append(img.save_derived(img.data, self.outfile(img.fname)))
        return ret

    def runcmd(self, cmd, logfile, raise_on_error=False):
        LOG.debug(cmd)
        with open(os.path.join(self.outdir, logfile), "w") as f:
            retval = subprocess.call(cmd, stdout=f, stderr=f)
        if retval != 0:
            msg = f" - Command {cmd} returned non-zero exit state {retval}"
            if raise_on_error:
                raise ModuleError(msg)
            else:
                LOG.warning(msg)
        return retval

    def bad_data(self, reason):
        raise ModuleError(f"Bad data: {reason}")

    def no_data(self, reason):
        raise ModuleError(f"Can't generate output - no input data: {reason}")

    def resample(
        self,
        src,
        tgt=None,
        is_roi=False,
        allow_rotated=False,
        tgt_affine=None,
        tgt_shape=None,
        cval=0.0,
    ):
        """
        Resample an image onto the grid from a target image or a specified grid affine/shape

        :return: nib.Nifti1Image
        """
        data_src = src.nii.get_fdata()
        while data_src.ndim < 3:
            data_src = data_src[..., np.newaxis]

        if tgt is not None:
            tgt_affine = tgt.affine
            data_tgt = tgt.nii.get_fdata()
            tgt_shape = data_tgt.shape
            tgt_header = tgt.nii.header
        elif tgt_affine is None or tgt_shape is None:
            raise ValueError("Need target image or affine/shape")
        else:
            tgt_header = None

        tgt_shape = list(tgt_shape)
        while len(tgt_shape) < 3:
            tgt_shape.append(1)
        tmatrix = np.dot(np.linalg.inv(tgt_affine), src.affine)
        tmatrix = np.linalg.inv(tmatrix)
        affine = tmatrix[:3, :3]
        offset = list(tmatrix[:3, 3])
        output_shape = list(tgt_shape[:3])

        if data_src.ndim == 4:
            # Make 4D affine with identity transform in 4th dimension
            affine = np.append(affine, [[0, 0, 0]], 0)
            affine = np.append(affine, [[0], [0], [0], [1]], 1)
            offset.append(0)
            output_shape.append(data_src.shape[3])

        LOG.debug(f"Resampling from\n{src.affine}\n{data_src.shape}")
        LOG.debug(f"To\n{tgt_affine}\n{tgt_shape}")
        LOG.debug(f"Net\n{affine}\n{offset}")
        same = (
            np.allclose(src.affine, tgt_affine)
            and list(data_src.shape[:3]) == output_shape[:3]
        )
        LOG.debug(f"Same: {same}")
        if same:
            return src.nii

        if not allow_rotated and not self.is_diagonal(affine):
            LOG.warn(
                "Data is rotated relative to segmentation - will not use this segmentation"
            )
            res_data = np.zeros(output_shape)
        else:
            LOG.debug(str(affine))
            LOG.debug(str(output_shape))
            LOG.debug(str(np.min(data_src)))
            LOG.debug(str(np.max(data_src)))
            res_data = scipy.ndimage.affine_transform(
                data_src,
                affine,
                offset=offset,
                output_shape=output_shape,
                order=0 if is_roi else 1,
                cval=cval,
                mode="grid-constant",
            )

        if is_roi:
            res_data = res_data.astype(np.uint32)
        return nib.Nifti1Image(res_data, tgt_affine, tgt_header)

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
            LOG.warn(
                f"Mask and image different shapes: {mask.shape}, {img.shape} - will not generate lightbox image {name}"
            )
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
                    bb_start, bb_end = max(0, bb_start - 5), min(
                        mask.shape[dim] - 1, bb_end + 5
                    )
                bbox[dim] = slice(bb_start, bb_end + 1)

            min_slice, max_slice = bbox[2].start, bbox[2].stop

        num_slices = max_slice - min_slice
        grid_size = int(math.ceil(math.sqrt(num_slices)))

        fig = Figure(figsize=(5, 5), dpi=200)
        FigureCanvas(fig)
        for slice_idx in range(num_slices):
            axes = fig.add_subplot(grid_size, grid_size, slice_idx + 1)
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
            axes.imshow(data_slice, cmap="gray")

            mask_slice = np.ma.masked_array(mask_slice, mask_slice == 0)
            axes.imshow(mask_slice, cmap="Reds", vmin=0, vmax=1)

            # Reverse y-axis so anterior is as the top
            axes.set_ylim(axes.get_ylim()[::-1])

            if self.pipeline.options.subjid and slice_idx == num_slices - 1:
                axes.text(
                    0,
                    0,
                    self.pipeline.options.subjid,
                    bbox={
                        "fill": True,
                        "edgecolor": "black",
                        "linewidth": 1,
                        "facecolor": "white",
                    },
                )

        fig.subplots_adjust(wspace=0, hspace=0.05)
        fig.savefig(self.outfile(name + ".png"), bbox_inches="tight")

    def matching_img(self, img, candidates, warn_none=False, warn_multiple=True):
        matches = []
        for candidate in candidates:
            if candidate.affine_matches(img):
                matches.append(candidate)
        if not matches:
            if warn_none:
                LOG.warn(
                    f" - Could not find matching image for {img.fname} in {candidates}"
                )
            return None
        elif len(matches) > 1:
            if warn_multiple:
                LOG.warn(
                    f" - Multiple matching images for {img.fname} in {candidates} - returning first which is {matches[0].fname}"
                )
        return matches[0]

    def blobs_by_size(self, seg_data, min_size=0):
        """
        :return blobs in a segmentation sorted by size, largest first
        """
        labelled = skimage.measure.label(seg_data)

        def _blob_size(blob):
            return np.sum(labelled == blob.label)

        blobs = list(skimage.measure.regionprops(labelled))
        blobs = sorted(blobs, key=_blob_size, reverse=True)
        blob_masks = []
        for blob in blobs:
            blob_mask = np.copy(seg_data)
            blob_mask[labelled != blob.label] = 0
            size = np.count_nonzero(blob_mask)
            if size > min_size:
                blob_masks.append(blob_mask)

        return blob_masks

    def run_nnunetv2(
        self,
        dataset_id,
        input_imgs,
        organ,
        data_name="",
        overlay_img=None,
        overlay_name=None,
    ):
        LOG.info(
            f" - Segmenting {organ.upper()} using {data_name} data in: {self.outdir}"
        )
        if any([img is None for img in input_imgs]):
            self.no_data(f"Missing input images for {organ} segmentation")

        for idx, img in enumerate(input_imgs):
            LOG.info(f" - Input {idx+1} from {img.fpath}")
            img.reorient2std().save(self.outfile(f"{organ}_000{idx}.nii.gz"))
        self.runcmd(
            [
                "nnUNetv2_predict",
                "-i",
                self.outdir,
                "-o",
                self.outdir,
                "-d",
                str(dataset_id),
                "-f",
                "all",
                "-c",
                "3d_fullres",
                "-device",
                "cpu",
            ],
            logfile="seg.log",
        )

        if overlay_img:
            seg = self.inimg(self.name, f"{organ}.nii.gz", src=self.OUTPUT)
            if not overlay_name:
                overlay_name = overlay_img.fname_noext
            LOG.info(
                f" - Generating overlay image of {organ.upper()} onto {overlay_name.upper()}"
            )
            self.lightbox(
                overlay_img, seg, name=f"{organ}_{overlay_name}_lightbox", tight=True
            )

    def split_lr(self, data, affine, side):
        LOG.debug(f" - Affine:\n{affine}")
        LOG.debug(f" - Shape: {data.shape}")
        axcodes = nib.orientations.aff2axcodes(affine)
        LOG.debug(f" - Axis codes: {axcodes}")
        lr_axis = axcodes.index("L") if "L" in axcodes else axcodes.index("R")
        lr_centre = data.shape[lr_axis] // 2
        split_data = np.copy(data)
        lr_slices = [slice(None)] * 3
        if side.lower() in ("l", "left"):
            lr_slices[lr_axis] = (
                slice(lr_centre)
                if "L" in axcodes
                else slice(lr_centre, data.shape[lr_axis])
            )
        elif side.lower() in ("r", "right"):
            lr_slices[lr_axis] = (
                slice(lr_centre)
                if "R" in axcodes
                else slice(lr_centre, data.shape[lr_axis])
            )
        else:
            raise RuntimeError(f" - Unknown side '{side}'")

        LOG.debug(f" - Slices: {lr_slices}")
        LOG.debug(f" - before non-zero: {np.count_nonzero(split_data)}")
        split_data[tuple(lr_slices)] = 0
        LOG.debug(f" - after non-zero: {np.count_nonzero(split_data)}")
        return (split_data > 0).astype(np.int32)


class CopyModule(Module):
    """
    A module which just copies an input file
    """

    def __init__(self, name, in_dir=None, in_name=None, out_name=None):
        Module.__init__(self, name)
        self.in_dir = in_dir if in_dir is not None else name
        self.in_name = in_name if in_name is not None else name
        self.out_name = out_name if out_name is not None else self.in_name

    def process(self):
        self.img = self.inimg(self.in_dir, f"{self.in_name}.nii.gz")
        LOG.info(f" - Copying {self.img.fname} - > {self.out_name}")
        self.img.save_derived(
            self.img.nii.get_fdata(), self.outfile(f"{self.out_name}.nii.gz")
        )
