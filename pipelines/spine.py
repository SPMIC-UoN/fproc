import logging

from fsort import ImageFile
from fproc.options import ArgumentParser
from fproc.pipeline import Pipeline
from fproc.module import Module, StatsModule

import numpy as np
import nibabel as nib
import skimage
import cv2
import torch
import pytorch_lightning as pl
import torch.nn.functional as functional
import segmentation_models_pytorch as smp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

class Regrid(Module):
    def __init__(self):
        Module.__init__(self, "regrid")

    def regrid(self, imgs):
        min_voxel_size, min_coord, max_coord = 1e20, [1e20, 1e20, 1e20], [-1e20, -1e20, -1e20]
        for img in imgs:
            voxel_sizes = img.nii.header.get_zooms()
            origin = img.affine[:3, 3]
            extent = np.dot(img.affine, [s - 1 for s in img.shape[:3]] + [1])[:3]
            LOG.info(f" - {img.fname}: from {origin} to {extent}")
            for dim in range(3):
                min_voxel_size = min(min_voxel_size, voxel_sizes[dim])
                min_coord[dim] = min(min_coord[dim], origin[dim])
                min_coord[dim] = min(min_coord[dim], extent[dim])
                max_coord[dim] = max(max_coord[dim], origin[dim])
                max_coord[dim] = max(max_coord[dim], extent[dim])
        LOG.info(f" - Minimum voxel dimension: {min_voxel_size}mm")
        LOG.info(f" - Grid from {min_coord} to {max_coord}")
        new_affine = np.zeros((4, 4), dtype=float)
        new_affine[:3, 3] = min_coord
        new_affine[3, 3] = 1.0
        for dim in range(3):
            new_affine[dim, dim] = min_voxel_size
        LOG.info(f" - New affine:\n{new_affine}")
        w2v = np.linalg.inv(new_affine)
        new_shape = [int(v) for v in np.dot(w2v, max_coord + [1.0])[:3]]
        LOG.info(f" - New shape: {new_shape}")

        # Convert axial in-plane aspect ratio to be close to UKB expected (224/162 ~= 1.4)
        ax_labels = nib.orientations.aff2axcodes(new_affine)
        LOG.info(f" - Axis mappings: {ax_labels}")
        if "R" in ax_labels:
            lr_axis = ax_labels.index("R")
        else:
            lr_axis = ax_labels.index("L")
        if "A" in ax_labels:
            pa_axis = ax_labels.index("A")
        else:
            pa_axis = ax_labels.index("P")
        pa_size = new_shape[lr_axis] / 1.4
        crop_voxels = int((new_shape[pa_axis] - pa_size) / 2)
        LOG.info(f" - Desired A-P dimension {pa_size}, actual {new_shape[pa_axis]}, cropping {crop_voxels} each side")
        crop = [0, 0, 0, 1]
        crop[pa_axis] = crop_voxels
        new_affine[:3, 3] = np.dot(new_affine, crop)[:3]
        new_shape[pa_axis] = new_shape[pa_axis] - 2*crop_voxels
        LOG.info(f" - New affine (cropped):\n{new_affine}")
        LOG.info(f" - New shape (cropped): {new_shape}")

        # Stitch together images. We crop top and bottom two slices as these often contain artefacts
        # and take the maximum value where there is overlap as there is generally signal dropout at the edges
        # FIXME assumes I-S axis is index 2
        output = np.ones(new_shape) * -999.0
        for img in imgs:
            cropped_data = img.data
            cropped_data[..., :2] = -999
            cropped_data[..., -2:] = -999
            img.save_derived(cropped_data, self.outfile("temp.nii.gz"))
            cropped_img = ImageFile(self.outfile("temp.nii.gz"))
            regridded = self.resample(cropped_img, allow_rotated=False, tgt_affine=new_affine, tgt_shape=new_shape, cval=-999.9)
            LOG.info(f" - Regridded {img.fname}")
            regridded_data = regridded.get_fdata()
            replace = np.logical_and(output < 0, regridded_data >= 0)
            average = np.logical_and(output >= 0, regridded_data >= 0)
            output[replace] = regridded_data[replace]
            output[average] = np.maximum(output[average], regridded_data[average])

        output[output < 0] = 0
        return nib.Nifti1Image(output, affine=new_affine)

    def process(self):
        fats = self.inimgs("dixon", "fat*.nii.gz")
        if not fats:
            self.no_data("No fat images found")
        fat_regrid = self.regrid(fats)
        fat_regrid.to_filename(self.outfile("fat.nii.gz"))

        waters = self.inimgs("dixon", "water*.nii.gz")
        if not waters:
            self.no_data("No water images found")
        water_regrid = self.regrid(waters)
        water_regrid.to_filename(self.outfile("water.nii.gz"))
        

class Spine_Model(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # Preprocessing parameters 
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Dice loss 
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # Normalize image 
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

class SpineSeg(Module):
    def __init__(self):
        Module.__init__(self, "spine_seg")

    def concatenate_adjacent(self, fat_tensor, slice_idx, context=1, augment=False):
        """
        Get the data slices above and below (n-1,n,n+1)

        :param fat_tensor: 3D volume, slices in dim 2
        :param slice_idx: Index of 'selected' slice
        :param context: number of adjacent slices, default = 1 
        """
        # get selected slice
        sel_slice = np.expand_dims(fat_tensor[..., slice_idx], 0)

        # augment
        if augment:
            sel_slice = np.fliplr(sel_slice).copy()

        # load slices before selected slice
        slices = []
        for sl in range(slice_idx - context, slice_idx + context + 1):
            # Ignore selected slice (see below).
            # If out of bounds, take selected slice, else load slice
            if sl == slice_idx:
                continue
            if sl < 0 or sl >= fat_tensor.shape[2]:
                s = sel_slice
            else:
                s = np.expand_dims(fat_tensor[..., sl], 0)
    
            slices.append(s)

        # concatenate all slices for context model learning
        # selected slice first because the network uses that one for the residual connection
        slices = [sel_slice] + slices
        return np.concatenate(slices, 0)

    def _reorient(self, data, ax_labels, reverse=False):
        if "R" in ax_labels:
            lr_axis, lr_flip = ax_labels.index("R"), False
        else:
            lr_axis, lr_flip = ax_labels.index("L"), True
        if "A" in ax_labels:
            pa_axis, pa_flip = ax_labels.index("A"), False
        else:
            pa_axis, pa_flip = ax_labels.index("P"), True
        if "S" in ax_labels:
            is_axis, is_flip = ax_labels.index("S"), False
        else:
            is_axis, is_flip = ax_labels.index("I"), True

        if not reverse:
            # Re-order to A-P, R-L, S-I axis order - this is what model expects
            # Size dimension of UK Biobank scan (156,224,501)
            data = data.transpose(pa_axis, lr_axis, is_axis)
            if not lr_flip: data = np.flip(data, lr_axis)
            if not pa_flip: data = np.flip(data, pa_axis)
            if not is_flip: data = np.flip(data, is_axis)
            return data
        else:
            # Invert the original->model reorientation
            axes = [-1, -1, -1]
            axes[pa_axis] = 0
            axes[lr_axis] = 1
            axes[is_axis] = 2
            data = data.transpose(axes)
            if not lr_flip: data = np.flip(data, lr_axis)
            if not pa_flip: data = np.flip(data, pa_axis)
            if not is_flip: data = np.flip(data, is_axis)
            return data

    def _regrid(self, tensor, size, batch_dims=0):
        is_numpy = isinstance(tensor, np.ndarray)
        if is_numpy:
            tensor = torch.Tensor(tensor)
        for i in range(2 - batch_dims):
            tensor = tensor.unsqueeze(0)
        tensor = functional.interpolate(tensor, size=size, mode='bilinear')
        for i in range(2 - batch_dims):
            tensor = tensor.squeeze(0)
        if is_numpy:
            return tensor.numpy()
        else:
            return tensor

    def _select_largest_blob_skimage(self, data):
        ret = np.copy(data)
        labels_mask = skimage.measure.label(data.astype(np.uint8))                       
        regions = skimage.measure.regionprops(labels_mask)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) > 1:
            for rg in regions[1:]:
                labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
        ret[labels_mask==0] = 0
        return ret

    def _select_largest_blob_cv2(self, data, return_contour=True, return_mask=True):
        contour_img = np.zeros(data.shape, np.uint8)
        mask = np.zeros(data.shape, np.uint8)
        contours, _ = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get contour with biggest area
            biggest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_img, [biggest_contour], -1, 255, cv2.FILLED)
            cv2.fillPoly(mask, pts=[biggest_contour], color=255)
        if return_contour and return_mask:
            return contour_img, mask
        elif return_contour:
            return contour_img
        elif return_mask:
            return mask

    def _save_coronal_data(self, data, name):
        data = data[:, np.newaxis, :]  # [IS, 1, 256]
        data_res = self._regrid(data, size=(self.AP, self.LR), batch_dims=1)  # [IS, AP, LR]
        data_res = self._reorient(np.transpose(data_res, (1, 2, 0)), self.ax_labels, reverse=True)
        LOG.debug(f" - Coronal image in original resolution: {data_res.shape}")
        self.fat.save_derived(data_res, self.outfile(f"{name}.nii.gz"))

    def _save_slicewise_volume_data(self, data, name):
        """
        :param data: Data array shape [SI, 128/AP, 256/LR]
        """
        data_res = self._regrid(data, size=(self.AP, self.LR), batch_dims=1)
        data_res = self._reorient(np.transpose(data_res, (1, 2, 0)), self.ax_labels, reverse=True)
        LOG.debug(f" - Stacked volume image in original resolution: {data_res.shape}")
        self.fat.save_derived(data_res, self.outfile(f"{name}.nii.gz"))

    def process(self):
        self.fat = self.inimg("regrid", "fat.nii.gz", is_depfile=True)
        fat_data = self.fat.data.astype(np.float32)
        fat_data = (fat_data - np.min(fat_data)) / (np.max(fat_data) - np.min(fat_data))
        LOG.info(f" - Normalized fat data from {self.fat.fname}")
        self.ax_labels = nib.orientations.aff2axcodes(self.fat.affine)
        LOG.info(f" - Axis mappings for fat image: {self.ax_labels}")

        # Re-order to A-P, R-L, S-I axis order - this is what model expects
        # Size dimension of UK Biobank scan (156,224,501)
        fat_data = self._reorient(fat_data, self.ax_labels)
        self.AP, self.LR, self.SI = tuple(fat_data.shape) 
        LOG.info(f" - Reoriented fat image: {fat_data.shape}")

        import torch
        import torch.nn.functional as functional
        torch.set_num_threads(10)
        fat_tensor = torch.Tensor(np.copy(fat_data)).float() # [AP, LR, IS]
        LOG.info(f" - Concatenating slices for {self.fat.fname}: {fat_data.shape}")
        thick_slices = []
        for i in range(fat_tensor.shape[2]):
            # Combine n-1,n and n+1 slices into a 'thick slice' [3, AP, LR]
            thick_slice = torch.Tensor(self.concatenate_adjacent(fat_tensor, i, 1))
            LOG.debug(f" - Concatenated thick slice {i} shape {thick_slice.shape}")
            # Convert to model resolution
            thick_slice = self._regrid(thick_slice, size=(128, 256), batch_dims=1)
            LOG.debug(f" - Interoplated slice {i} shape {thick_slice.shape}")
            thick_slices.append(thick_slice.numpy())  # [3, 128, 256]

        ## Segmentation
        LOG.info(f" - Segmenting {len(thick_slices)} slices")
        # Load trained model in evaluation mode
        model = Spine_Model("deeplabv3plus", "resnet34", in_channels=3, out_classes=1)
        model.load_state_dict(torch.load('weights/resnet34_deepV3.pth', map_location=torch.device('cpu')))
        model.eval()

        # Run segmentation
        inputs, probs, masks, main_contours, main_masks = [], [], [], [], []
        for idx, thick_slice in enumerate(thick_slices):
            LOG.debug(f" - Segmenting thick slice {idx}: {thick_slice.shape}")
            inputs.append(thick_slice[0, ...])  # [128, 256]

            with torch.no_grad():
                model.eval()
                logits = model(torch.Tensor(thick_slice))

            prob_mask = logits.sigmoid() # [128, 256]
            probs.append(prob_mask.numpy().squeeze())

            # Threshold mask and predictions
            thresh_mask = prob_mask.numpy().squeeze() * 255
            mask = thresh_mask.astype('uint8')
            LOG.debug(f" - Mask {mask.shape}")
            masks.append(mask)

            # Select biggest slicewise blob
            slice_main_contour, slice_main_mask = self._select_largest_blob_cv2(mask)
            main_contours.append(slice_main_contour)
            main_masks.append(slice_main_mask)

        # Stack the axial slices vertically to reconstruct the volume
        # Convert back to original resolution and orientation and save
        volume_prob = np.stack(probs, axis=0)  # [IS, 128, 256]
        volume_mask = np.stack(masks, axis=0)  # [IS, 128, 256]
        volume_main_contours = np.stack(main_contours, axis=0)  # [IS, 128, 256]
        volume_main_masks = np.stack(main_masks, axis=0)  # [IS, 128, 256]
        volume_inputs = np.stack(inputs, axis=0)
        LOG.info(f" - Stacked slicewise segmentation data: {volume_prob.shape}")
        self._save_slicewise_volume_data(volume_prob, "prob")
        self._save_slicewise_volume_data(volume_mask, "mask")
        self._save_slicewise_volume_data(volume_main_contours, "seg_contours")
        self._save_slicewise_volume_data(volume_main_masks, "masks_cleaned")
        self._save_slicewise_volume_data(volume_inputs, "input")

        # Compute coronal masks for cleaning, visualisation and curvature calculation
        coronal_mask1 = np.max(volume_main_masks, axis=1) # [IS, 256]
        self._save_coronal_data(coronal_mask1, "coronal_mask1")

        # Compute coronal masks for cleaning, visualisation and curvature calculation
        coronal_mask = np.mean(volume_main_contours, axis=1) # [IS, 256]
        LOG.info(f" - Coronal mask: {np.min(coronal_mask)}, {np.max(coronal_mask)}: {coronal_mask.shape}")
        coronal_mask_normed = (coronal_mask/np.max(coronal_mask)*255).astype(np.uint8)
        self._save_coronal_data(coronal_mask_normed, "coronal_mask_normed")
        LOG.info(f" - Normed coronal mask vox={np.count_nonzero(coronal_mask_normed)}")
        coronal_mask_bin = (coronal_mask_normed > 0).astype(np.uint8)
        self._save_coronal_data(coronal_mask_bin, "coronal_mask_bin")
        LOG.info(f" - Binarised coronal mask vox={np.count_nonzero(coronal_mask_bin)}")
        
        # Clean coronal mask to only include blobs with area > 1000
        contours, _ = cv2.findContours(coronal_mask_normed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        LOG.info(f" - Found {len(contours)} coronal contours")
        selected_contours = []
        coronal_mask_clean = np.copy(coronal_mask_bin).astype(np.uint8)  # [IS, 256]        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:
                LOG.info(f" - Contour {idx} has area {area}")
                selected_contours.append(contour)
                cv2.fillPoly(coronal_mask_clean, pts=selected_contours, color=255)  # [IS, 256, 3]
                LOG.info(f" - Filled coronal mask vox={np.count_nonzero(coronal_mask_clean)}")
        LOG.info(f" - Selected {len(selected_contours)} coronal contours with area > 1000")
        self._save_coronal_data(coronal_mask_clean, "coronal_mask_clean")

        # Erode coronal mask
        kernel = np.ones((5, 5), np.uint8) 
        coronal_mask_ero = cv2.erode(coronal_mask_clean, kernel)
        LOG.info(f" - Eroded coronal mask vox={np.count_nonzero(coronal_mask_ero)}")
        self._save_coronal_data(coronal_mask_ero, "coronal_mask_ero")

        # Back to original resolution
        coronal_mask_res = self._regrid(coronal_mask_ero, size=(self.SI, self.LR))   
        LOG.info(f" - Interpolated coronal mask vox={np.count_nonzero(coronal_mask_res)}")
        self._save_coronal_data(coronal_mask_res, "coronal_mask_res")

        # Remove any small blobs generated by erosion or regridding
        coronal_mask_clean_final = self._select_largest_blob_skimage(coronal_mask_res)
        self._save_coronal_data(coronal_mask_clean_final, "coronal_mask_clean_final")

        # Curvature of Spine Calculation
        #
        # 1. Fit a cubic spline through midpoint extracted from contour of spine 
        #    on axial slices 
        # 2. Compute maximum curvature along the spine
        slice_have_spine = np.max(coronal_mask_clean_final, axis=1)
        top = np.argmax(slice_have_spine)
        bottom = len(slice_have_spine) - np.argmax(slice_have_spine[::-1]) - 1
        LOG.info(f" - Finding centroids of mask from slice {top} to {bottom}")
        st = []
        centroid_img = np.zeros(fat_data.shape, dtype=np.uint8)  # [AP, LR, IS]
        for i in range(top,bottom):
            inp = volume_main_contours[i,:,:]  # [128, 256]
            LOG.debug(f" - Slice mask: {inp.shape}")

            # Back to original resolution 
            inp = self._regrid(inp, size=(fat_tensor.shape[0],fat_tensor.shape[1]))  # [PA, LR]
            inp = (inp * 255).astype(np.uint8)
            LOG.debug(f" - Interpolated slice mask: {inp.shape}")

            # Apply cv2.threshold() to get a binary image
            ret, thresh = cv2.threshold(inp, 20, 255, cv2.THRESH_BINARY)

            # Calculate centroid of the mask
            # Find contours: FIXME this appears to be unused
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            try:
                M = cv2.moments(thresh)
                Mx = int(M['m10'] / M['m00'])
                My = int(M['m01'] / M['m00'])
                centroid_img[My, Mx, i] = 1

                # Draw a circle based centered at centroid coordinates FIXME unused
                cent = cv2.circle(np.ascontiguousarray(inp, dtype=np.uint8), (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 1, 255, -1)
                c = [Mx, i]
                st.append(c)
            except:
                LOG.warn(f"Failed to find centroid for slice {i}")
        centroid_img = self._reorient(centroid_img, self.ax_labels, reverse=True)
        self.fat.save_derived(centroid_img, self.outfile("centroids.nii.gz"))

        LOG.info(f" - Fitting cubic spline to centroids")
        st = np.vstack(st)
        # FIXME should already be sorted?
        st = sorted(st, key=lambda x: x[1], reverse=False)
        x = [t[0] for t in st]
        y = [t[1] for t in st]
        f = CubicSpline(y, x, bc_type='natural')
        y_new = np.linspace(min(y), max(y), 30)
        x_new = f(y_new)

        # Curvature computation taking the second derivatives
        dx_dt = np.gradient(x_new)
        dy_dt = np.gradient(y_new)
        ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        d2s_dt2 = np.gradient(ds_dt)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        LOG.info(f" - Curvature: {curvature}")
        LOG.info(f" - Max curvature: {np.max(curvature)}.3f")
        with open(self.outfile("curvature.csv"), "w") as f:
            for y, curv in zip(y_new, curvature):
                f.write("%i, %.4f" % (y, curv))

        # Generate results output curvature value
        bg = fat_tensor[round(fat_tensor.shape[0]/2), :, :].numpy()  # [1, LR, IS]
        plt.imshow(bg.T, cmap='gray')
        plt.imshow(coronal_mask_clean_final, cmap='jet', alpha=0.5)
        plt.scatter(x_new, y_new, s=0.1, marker='x', c='white')
        plt.axis("off")
        plt.savefig(self.outfile("coronal_overlay.png"), bbox_inches = 'tight', pad_inches=0, dpi=1000)
        plt.close()
        LOG.info(f" - Generated coronal overlay image in coronal_overlay.png")

MODULES = [
    Regrid(),
    SpineSeg(),
]

class SpineArgumentParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, "spine", __version__)
        
class Spine(Pipeline):
    def __init__(self):
        Pipeline.__init__(self, "spine", __version__, SpineArgumentParser(), MODULES)

if __name__ == "__main__":
    Spine().run()
