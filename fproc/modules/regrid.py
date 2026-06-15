"""
FPROC: Modules for regridding data
"""

import logging
import os

import nibabel as nib
import numpy as np

from fsort.image_file import ImageFile
from fproc.module import Module

LOG = logging.getLogger(__name__)


class Stitch(Module):
    def __init__(self, name="stitch", **kwargs):
        self._img_dir = kwargs.get("img_dir", None)
        Module.__init__(self, name, deps=[self._img_dir], **kwargs)

    def regrid(self, imgs, norm=False):
        # Determine the full extent of the grid needed to cover all images
        min_voxel_size, min_coord, max_coord = (
            [1e20, 1e20, 1e20],
            [1e20, 1e20, 1e20],
            [-1e20, -1e20, -1e20],
        )
        imgs = [img.reorient2std() for img in imgs]
        for img in imgs:
            voxel_sizes = img.nii.header.get_zooms()
            origin = img.affine[:3, 3]
            extent = np.dot(img.affine, [s - 1 for s in img.shape[:3]] + [1])[:3]
            LOG.info(f" - {img.fname}: from {origin} to {extent}")
            for dim in range(3):
                min_voxel_size[dim] = min(min_voxel_size[dim], voxel_sizes[dim])
                min_coord[dim] = min(min_coord[dim], origin[dim])
                min_coord[dim] = min(min_coord[dim], extent[dim])
                max_coord[dim] = max(max_coord[dim], origin[dim])
                max_coord[dim] = max(max_coord[dim], extent[dim])

        # For now we take voxel size from first image to construct the affine
        min_voxel_size = imgs[0].nii.header.get_zooms()
        LOG.info(f" - Voxel sizes: {min_voxel_size}mm")
        LOG.info(f" - Grid from {min_coord} to {max_coord}")
        stitched_affine = np.zeros((4, 4), dtype=float)
        stitched_affine[:3, 3] = min_coord
        stitched_affine[3, 3] = 1.0
        for dim in range(3):
            stitched_affine[dim, dim] = min_voxel_size[dim]
        LOG.info(f" - Stitched affine:\n{stitched_affine}")

        w2v = np.linalg.inv(stitched_affine)
        stitched_shape = [int(round(v + 1)) for v in np.dot(w2v, max_coord + [1.0])[:3]]
        LOG.info(f" - Stitched shape: {stitched_shape}")

        # Prepare stitching by creating output array and determining intensity range for normalisation if needed
        nvols = None
        crop_slices = self.kwargs.get("crop_slices", 0)
        stitch_axis = self.kwargs.get("stitch_axis", 2)
        nostitch_axes = [0, 1, 2]
        nostitch_axes.remove(stitch_axis)
        nostitch_axes = tuple(nostitch_axes)
        intensity_range = None
        for idx, img in enumerate(imgs):
            img_data = img.data
            while len(img_data.shape) < 4:
                img_data = np.expand_dims(img_data, axis=-1)
            if nvols is None:
                # Prepare the output array - each chunk will be added to this with overlap averaging
                stitched_shape.append(img_data.shape[3])
                nvols = img_data.shape[3]
                output = np.ones(stitched_shape) * -999.0
                intensity_range = [[1e20, -1e20] for _ in range(nvols)]
            else:
                if img_data.shape[3] != nvols:
                    self.bad_data(
                        f"Image {img.fname} has {img_data.shape[3]} volumes but expected {nvols} based on previous images - cannot stitch"
                    )

            if norm:
                # If we are normalising, update the per-volume intensity range
                for vol in range(nvols):
                    pc1, pc99 = np.percentile(img_data[..., vol], 1), np.percentile(
                        img_data[..., vol], 99
                    )
                    LOG.info(
                        f" - Image {img.fname} vol {vol} has values from {pc1} to {pc99}"
                    )
                    if pc1 < intensity_range[vol][0]:
                        intensity_range[vol][0] = pc1
                    if pc99 > intensity_range[vol][1]:
                        intensity_range[vol][1] = pc99

        # Stitch together images. We crop top and bottom two slices as these often contain artefacts
        # and take the maximum value where there is overlap as there is generally signal dropout at the edges
        debug_output = self.kwargs.get("debug_output", False)
        for idx, img in enumerate(imgs):
            img_data = img.data
            while len(img_data.shape) < 4:
                img_data = np.expand_dims(img_data, axis=-1)

            if norm:
                for vol in range(nvols):
                    p1, p99 = np.percentile(img_data[..., vol], 1), np.percentile(
                        img_data[..., vol], 99
                    )
                    t1, t99 = intensity_range[vol]
                    LOG.info(
                        f" - Normalising {img.fname} vol {vol} from range {p1}-{p99} to {t1}-{t99}"
                    )
                    if p1 == p99:
                        LOG.warning(
                            f" - Image {img.fname} vol {vol} has zero intensity range - skipping normalisation"
                        )
                        continue
                    img_data[..., vol] = (img_data[..., vol] - p1) * (t99 - t1) / (
                        p99 - p1
                    ) + t1

            if crop_slices:
                LOG.info(
                    f" - Cropping {crop_slices} slices from {img.fname} to remove artefacts"
                )
                print(img_data.shape)
                slices = [slice(None)] * 4
                slices[stitch_axis] = slice(0, crop_slices)
                if idx != 0:
                    img_data[tuple(slices)] = -999
                slices[stitch_axis] = slice(-crop_slices, None)
                if idx != len(imgs) - 1:
                    img_data[tuple(slices)] = -999

            img.save_derived(
                img_data, self.outfile(f"{img.fname_noext}_postcrop.nii.gz")
            )
            cropped_img = ImageFile(self.outfile(f"{img.fname_noext}_postcrop.nii.gz"))
            regridded = self.resample(
                cropped_img,
                allow_rotated=True,
                tgt_affine=stitched_affine,
                tgt_shape=stitched_shape,
                cval=-999.9,
            )
            regridded_data = regridded.get_fdata()
            LOG.info(
                f" - Regridded {img.fname}: new shape {regridded_data.shape}, {np.sum(regridded_data == -999.9)} voxels with no data"
            )

            # Overlap averaging, FIXME assuming dim 2 is stitch dim
            # We define two weighting regions, one where we only have new data (replace existing fill values)
            # and one where we have data > 0 in both existing and new - these will be averaged with a linear
            # weighting across the overlap region in the Z dimension
            weighting_new = np.zeros_like(output)
            replace = np.logical_and(output <= 0, regridded_data > 0)
            average = np.logical_and(output > 0, regridded_data > 0)

            if np.any(replace):
                replace_slices = np.any(replace, axis=nostitch_axes)
                indices_replace = np.where(replace_slices)[0]
                replace_min, replace_max = indices_replace.min(), indices_replace.max()
                replace_slices = [slice(None)] * 4
                replace_slices[stitch_axis] = slice(replace_min, replace_max + 1)
                weighting_new[tuple(replace_slices)] = 1.0

            if np.any(average):
                average_z = np.any(average, axis=nostitch_axes)
                indices_z = np.where(average_z)[0]
                average_z_min, average_z_max = indices_z.min(), indices_z.max()
                LOG.info(
                    f" - Averaging overlap from slices {average_z_min} to {average_z_max}"
                )
                average_slices = [slice(None)] * 4
                average_slices[stitch_axis] = slice(average_z_min, average_z_max + 1)
                weighting_new[tuple(average_slices)] = np.linspace(
                    0, 1, average_z_max - average_z_min + 1
                )[:, np.newaxis]

            weighting_cur = 1 - weighting_new
            if debug_output:
                output_thischunk = regridded_data * weighting_new
                nii_thischunk = nib.Nifti1Image(
                    output_thischunk, affine=stitched_affine
                )
                nii_thischunk.to_filename(
                    self.outfile(f"{img.fname_noext}_chunk.nii.gz")
                )
                output_prevchunk = output * weighting_cur
                nii_prevchunk = nib.Nifti1Image(
                    output_prevchunk, affine=stitched_affine
                )
                nii_prevchunk.to_filename(
                    self.outfile(f"{img.fname_noext}_chunk_prev.nii.gz")
                )
            else:
                os.remove(self.outfile(f"{img.fname_noext}_postcrop.nii.gz"))

            output = output * weighting_cur + regridded_data * weighting_new

        output[output < 0] = 0
        return nib.Nifti1Image(output, affine=stitched_affine)

    def process(self):
        if not self._img_dir:
            self.no_data(f"No images dir specified")
        img_src = self.kwargs.get("img_src", self.OUTPUT)

        for img_glob, out_fname in self.kwargs.get("imgs", {}).items():
            imgs = self.inimgs(self._img_dir, img_glob, src=img_src)
            if not imgs:
                LOG.warn(
                    f"No images found in {self._img_dir} matching {img_glob} - ignoring this set"
                )
                continue

            elif len(imgs) == 1:
                LOG.info(
                    f" - One image found matching {img_glob} - copying to {out_fname} without stitching"
                )
                imgs[0].save(self.outfile(out_fname))
                continue

            norm = self.kwargs.get("normalise", False)

            LOG.info(
                f" - Stitching slice images from {self._img_dir}/{img_glob} - {len(imgs)} images found"
            )

            imgs_regrid = self.regrid(imgs, norm)
            LOG.info(f" - Saving to {out_fname}")
            imgs_regrid.to_filename(self.outfile(out_fname))


class StitchSlices(Module):
    def __init__(self, name="stitch", **kwargs):
        self._img_dir = kwargs.get("img_dir", None)
        Module.__init__(self, name, deps=[self._img_dir], **kwargs)

    def process(self):
        if not self._img_dir:
            self.no_data(f"No images dir specified")
        img_src = self.kwargs.get("img_src", self.OUTPUT)

        for img_glob, out_fname in self.kwargs.get("imgs", {}).items():
            imgs = self.inimgs(self._img_dir, img_glob, src=img_src)
            if not imgs:
                LOG.warn(
                    f"No images found in {self._img_dir} matching {img_glob} - ignoring this set"
                )
                continue

            elif len(imgs) == 1:
                LOG.info(
                    f" - One image found matching {img_glob} - copying to {out_fname} without stitching"
                )
                imgs[0].save(self.outfile(out_fname))
                continue

            LOG.info(
                f" - Stitching slice images from {self._img_dir}/{img_glob} - {len(imgs)} images found"
            )

            flat_dim = []
            affine_tol = self.kwargs.get("affine_tol", 1e-3)
            origins = []
            trans = None
            ignore = False
            for img in imgs:
                if trans is None:
                    trans = img.affine[:3, :3]
                elif not np.allclose(img.affine[:3, :3], trans, atol=affine_tol):
                    LOG.warn(
                        f"Images have different orientations: {trans} vs {img.affine[:3, :3]} - ignoring this set"
                    )
                    ignore = True
                    break
                try:
                    flat_dim.append(list(img.shape).index(1))
                except ValueError:
                    LOG.warn(
                        f"Image {img.fname} does not have a since-slice dimension (shape {img.shape}) - ignoring this set"
                    )
                    ignore = True
                    break
                origins.append(img.affine[:3, 3])

            if ignore:
                continue

            if len(set(flat_dim)) > 1:
                LOG.warn(
                    f"Images have different slice dimensions: {flat_dim} - ignoring this set"
                )
                continue

            flat_dim = flat_dim[0]
            new_shape = list(imgs[0].shape)
            new_shape[flat_dim] = len(imgs)
            LOG.info(f" - Slice dimension {flat_dim} - new shape will be {new_shape}")

            # Determine the order of the slices
            w2v = np.linalg.inv(trans)
            slice_normal = trans[flat_dim]
            slice_order = []
            for idx, img in enumerate(imgs):
                slice_order.append(np.dot(w2v, origins[idx] - origins[0])[flat_dim])
                LOG.info(f" - Image {img.fname} slice order {slice_order[-1]}")

            sorted_imgs = [img for _, img in sorted(zip(slice_order, imgs))]
            output = np.zeros(new_shape)
            for idx, img in enumerate(sorted_imgs):
                sl = [slice(None)] * 3
                sl[flat_dim] = idx
                output[tuple(sl)] = np.squeeze(img.data, axis=flat_dim)

            affine = sorted_imgs[0].affine
            nii = nib.Nifti1Image(
                output, affine=affine, header=sorted_imgs[0].nii.header
            )
            nii.to_filename(self.outfile(out_fname))


class CombineSegs(Module):
    """
    Combine multiple segmentation files onto a common high-resolution grid
    """

    def __init__(self, name="combine_segs", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        seg_dirs = self.kwargs.get("seg_dirs", [])
        if not seg_dirs:
            self.no_data("No segmentation directories specified")

        seg_globs = self.kwargs.get("seg_globs", [])
        if not seg_globs:
            seg_globs = ["*.nii.gz"] * len(seg_dirs)
        elif len(seg_globs) == 1:
            seg_globs = seg_globs * len(seg_dirs)
        elif len(seg_globs) != len(seg_dirs):
            self.no_data(
                f"Number of seg_globs ({len(seg_globs)}) must match seg_dirs ({len(seg_dirs)})"
            )

        seg_src = self.kwargs.get("seg_src", self.OUTPUT)

        # Collect all segmentations from all directories
        segs = []
        for seg_dir, seg_glob in zip(seg_dirs, seg_globs):
            dir_segs = self.inimgs(seg_dir, seg_glob, src=seg_src)
            if dir_segs:
                LOG.info(
                    f" - Found {len(dir_segs)} segmentations in {seg_dir}/{seg_glob}"
                )
                segs.extend(dir_segs)
            else:
                LOG.warn(f" - No segmentations found in {seg_dir}/{seg_glob}")

        if not segs:
            self.no_data(f"No segmentations found in any of the specified directories")

        LOG.info(f" - Combining {len(segs)} segmentations total")

        # Find minimum voxel size (maximum resolution) and FOV that covers all segmentations
        min_voxel_size = [1e20, 1e20, 1e20]
        min_coord = [1e20, 1e20, 1e20]
        max_coord = [-1e20, -1e20, -1e20]

        for seg in segs:
            voxel_sizes = seg.nii.header.get_zooms()[:3]
            corners = []
            # Get all 8 corners of the volume
            for i in [0, seg.shape[0] - 1]:
                for j in [0, seg.shape[1] - 1]:
                    for k in [0, seg.shape[2] - 1]:
                        corner = np.dot(seg.affine, [i, j, k, 1])[:3]
                        corners.append(corner)
            print("affine\n", seg.affine)
            print("shape\n", seg.shape)
            print("corners\n", corners)
            # Update min voxel size and min/max coordinates
            for dim in range(3):
                min_voxel_size[dim] = min(min_voxel_size[dim], voxel_sizes[dim])
                min_coord[dim] = min(min_coord[dim], min(c[dim] for c in corners))
                max_coord[dim] = max(max_coord[dim], max(c[dim] for c in corners))

        LOG.info(f" - Minimum voxel size (max resolution): {min_voxel_size} mm")
        LOG.info(f" - FOV from {min_coord} to {max_coord}")

        # Create new affine matrix
        new_affine = np.eye(4)
        new_affine[:3, 3] = min_coord
        for dim in range(3):
            new_affine[dim, dim] = min_voxel_size[dim]

        # Calculate new shape
        extent = np.array(max_coord) - np.array(min_coord)
        new_shape = [
            int(np.ceil(extent[dim] / min_voxel_size[dim])) + 1 for dim in range(3)
        ]

        LOG.info(f" - New grid shape: {new_shape}")
        LOG.info(f" - New affine:\n{new_affine}")

        # Combine segmentations on the new grid
        output = np.zeros(new_shape, dtype=np.int16)

        for idx, seg in enumerate(segs):
            LOG.info(f" - Resampling {seg.fname}")
            resampled = self.resample(
                seg,
                allow_rotated=True,
                tgt_affine=new_affine,
                tgt_shape=new_shape,
                is_roi=True,
            )
            resampled_data = resampled.get_fdata().astype(np.int16)

            # Combine: take maximum value where overlap occurs
            # This assumes segmentation labels don't conflict
            output = np.maximum(output, resampled_data)

        out_fname = self.kwargs.get("out_fname", "combined_segs.nii.gz")
        LOG.info(f" - Saving combined segmentation to {out_fname}")
        nii = nib.Nifti1Image(output, affine=new_affine)
        nii.to_filename(self.outfile(out_fname))
