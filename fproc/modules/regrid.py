"""
FPROC: Modules for regridding data
"""
import logging

import nibabel as nib
import numpy as np

from fsort.image_file import ImageFile
from fproc.module import Module

LOG = logging.getLogger(__name__)

class Stitch(Module):
    def __init__(self, name="stitch", **kwargs):
        Module.__init__(self, name, **kwargs)

    def regrid(self, imgs):
        min_voxel_size, min_coord, max_coord = [1e20, 1e20, 1e20], [1e20, 1e20, 1e20], [-1e20, -1e20, -1e20]
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
        min_coord = imgs[0].affine[:3, 3]
        min_voxel_size = imgs[0].nii.header.get_zooms()
        LOG.info(f" - Minimum voxel dimension: {min_voxel_size}mm")
        LOG.info(f" - Grid from {min_coord} to {max_coord}")
        #new_affine = np.zeros((4, 4), dtype=float)
        #new_affine[:3, 3] = min_coord
        #new_affine[3, 3] = 1.0
        #for dim in range(3):
        #    new_affine[dim, dim] = min_voxel_size[dim]
        new_affine = imgs[0].affine
        new_affine[:3, 3] = min_coord
        LOG.info(f" - New affine:\n{new_affine}")
        w2v = np.linalg.inv(new_affine)
        #new_shape = [int(v) for v in np.dot(w2v, max_coord + [1.0])[:3]]
        LOG.info(f" - New shape: {new_shape}")

        # Stitch together images. We crop top and bottom two slices as these often contain artefacts
        # and take the maximum value where there is overlap as there is generally signal dropout at the edges
        # FIXME assumes I-S axis is index 2
        output = np.ones(new_shape) * -999.0
        for img in imgs:
            cropped_data = img.data
            #cropped_data[..., :2] = -999
            #cropped_data[..., -2:] = -999
            img.save_derived(cropped_data, self.outfile("temp.nii.gz"))
            cropped_img = ImageFile(self.outfile("temp.nii.gz"))
            regridded = self.resample(cropped_img, allow_rotated=True, tgt_affine=new_affine, tgt_shape=new_shape, cval=-999.9)
            LOG.info(f" - Regridded {img.fname}")
            regridded_data = regridded.get_fdata()
            replace = np.logical_and(output < 0, regridded_data >= 0)
            average = np.logical_and(output >= 0, regridded_data >= 0)
            output[replace] = regridded_data[replace]
            output[average] = np.maximum(output[average], regridded_data[average])

        output[output < 0] = 0
        return nib.Nifti1Image(output, affine=new_affine)

    def process(self):
        img_dir = self.kwargs.get("img_dir", None)
        if not img_dir:
            self.no_data(f"No images dir specified")

        img_glob = self.kwargs.get("img_glob", "*.nii.gz")
        img_src = self.kwargs.get("img_src", self.OUTPUT)
        imgs = self.inimgs(img_dir, img_glob, src=img_src)
        if not imgs:
            self.no_data(f"No images found in {img_dir} matching {img_glob}")

        LOG.info(f" - Regridding images from {img_dir}/{img_glob} - {len(imgs)} images found")
        imgs_regrid = self.regrid(imgs)
        out_fname = self.kwargs.get("out_fname", "regrid.nii.gz")
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
                LOG.warn(f"No images found in {self._img_dir} matching {img_glob} - ignoring this set")
                continue

            elif len(imgs) == 1:
                LOG.info(f" - One image found matching {img_glob} - copying to {out_fname} without stitching")
                imgs[0].save(self.outfile(out_fname))
                continue

            LOG.info(f" - Stitching slice images from {self._img_dir}/{img_glob} - {len(imgs)} images found")
            flat_dim = []
            affine_tol = self.kwargs.get("affine_tol", 1e-3)
            origins = []
            trans = None
            ignore = False
            for img in imgs:
                if trans is None:
                    trans = img.affine[:3, :3]
                elif not np.allclose(img.affine[:3, :3], trans, atol=affine_tol):
                    LOG.warn(f"Images have different orientations: {trans} vs {img.affine[:3, :3]} - ignoring this set")
                    ignore = True
                    break
                try:
                    flat_dim.append(list(img.shape).index(1))
                except ValueError:
                    LOG.warn(f"Image {img.fname} does not have a since-slice dimension (shape {img.shape}) - ignoring this set")
                    ignore = True
                    break
                origins.append(img.affine[:3, 3])

            if ignore:
                continue

            if len(set(flat_dim)) > 1:
                LOG.warn(f"Images have different slice dimensions: {flat_dim} - ignoring this set")
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
            nii = nib.Nifti1Image(output, affine=affine, header=sorted_imgs[0].nii.header)
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
            self.no_data(f"Number of seg_globs ({len(seg_globs)}) must match seg_dirs ({len(seg_dirs)})")
        
        seg_src = self.kwargs.get("seg_src", self.OUTPUT)
        
        # Collect all segmentations from all directories
        segs = []
        for seg_dir, seg_glob in zip(seg_dirs, seg_globs):
            dir_segs = self.inimgs(seg_dir, seg_glob, src=seg_src)
            if dir_segs:
                LOG.info(f" - Found {len(dir_segs)} segmentations in {seg_dir}/{seg_glob}")
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
            for i in [0, seg.shape[0]-1]:
                for j in [0, seg.shape[1]-1]:
                    for k in [0, seg.shape[2]-1]:
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
        new_shape = [int(np.ceil(extent[dim] / min_voxel_size[dim])) + 1 for dim in range(3)]
        
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
                is_roi=True
            )
            resampled_data = resampled.get_fdata().astype(np.int16)
            
            # Combine: take maximum value where overlap occurs
            # This assumes segmentation labels don't conflict
            output = np.maximum(output, resampled_data)
        
        out_fname = self.kwargs.get("out_fname", "combined_segs.nii.gz")
        LOG.info(f" - Saving combined segmentation to {out_fname}")
        nii = nib.Nifti1Image(output, affine=new_affine)
        nii.to_filename(self.outfile(out_fname))
