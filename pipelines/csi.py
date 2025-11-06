"""
Processing pipeline for CSI spectroscopy data
"""

import logging
import os

import nibabel as nib
import numpy as np
from fsort.image_file import ImageFile
from nibabel import processing

from fproc.module import Module

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)


class StrucPreproc(Module):
    """
    Preprocess MPRAGE structural images

     - bias correction
     - brain extraction
    """

    def __init__(self):
        Module.__init__(self, "struc_preproc")

    def process(self):
        mprages = self.inimgs("mprage", "mprage*.nii.gz")
        if not mprages:
            self.no_data("No MPRAGE images found")

        for mprage in mprages:
            LOG.info(f" - Processing MPRAGE: {mprage.fname}")
            mprage.save(self.outfile(mprage.fname))

            LOG.info("   - Running ANTs N4 bias correction")
            mprage_biascorr = self.outfile(
                mprage.fname.replace(".nii.gz", "_biascorr.nii.gz")
            )
            biascorr_cmd = [
                "N4BiasFieldCorrection",
                "-d",
                "3",
                "-i",
                mprage.fpath,
                "-o",
                mprage_biascorr,
            ]
            LOG.debug(biascorr_cmd)
            self.runcmd(
                biascorr_cmd,
                mprage.fname.replace(".nii.gz", "_biascorr.log"),
                raise_on_error=True,
            )

            LOG.info("   - Running mri_synthstrip (brain extraction)")
            mprage_brain = self.outfile(
                mprage.fname.replace(".nii.gz", "_brain.nii.gz")
            )
            synthstrip_cmd = [
                "mri_synthstrip",
                "-i",
                mprage_biascorr,
                "-o",
                mprage_brain,
            ]
            LOG.debug(synthstrip_cmd)
            self.runcmd(
                synthstrip_cmd,
                mprage.fname.replace(".nii.gz", "_synthstrip.log"),
                raise_on_error=True,
            )


class StrucReg(Module):
    """
    Register structural images to MNI space using ANTs
    """

    def __init__(self):
        Module.__init__(self, "struc_reg")

    def process(self):
        struc_brain_imgs = self.inimgs("struc_preproc", "*_brain.nii.gz")
        if not struc_brain_imgs:
            self.no_data("No brain-extracted structural images found")

        fsldir = os.environ.get("FSLDIR", None)
        if not fsldir:
            raise RuntimeError("FSLDIR environment variable not set")

        for struc_brain in struc_brain_imgs:
            LOG.info(
                f" - Processing brain-extracted structural image: {struc_brain.fname}"
            )

            LOG.info("   - Running ANTs registration to MNI template")
            ants_prefix = self.outfile(struc_brain.fname.replace(".nii.gz", "_"))
            ants_cmd = [
                "antsRegistrationSyN.sh",
                "-d",
                "3",
                "-f",
                f"{fsldir}/data/standard/MNI152_T1_2mm_brain.nii.gz",
                "-m",
                struc_brain.fpath,
                "-o",
                ants_prefix,
            ]
            LOG.debug(ants_cmd)
            self.runcmd(
                ants_cmd,
                struc_brain.fname.replace(".nii.gz", "_ants.log"),
                raise_on_error=True,
            )

            LOG.info("   - Applying ANTs transforms to original MPRAGE")
            ants_warp = ants_prefix + "1Warp.nii.gz"
            ants_affine = ants_prefix + "0GenericAffine.mat"
            struc_orig = ImageFile(
                struc_brain.fpath.replace("_brain.nii.gz", ".nii.gz")
            )
            mprage_mni_output = self.outfile(
                struc_orig.fname.replace(".nii.gz", "_mni.nii.gz")
            )
            applytransforms_cmd = [
                "antsApplyTransforms",
                "-d",
                "3",
                "-i",
                struc_orig.fpath,
                "-r",
                f"{fsldir}/data/standard/MNI152_T1_2mm_brain.nii.gz",
                "-t",
                ants_warp,
                "-t",
                ants_affine,
                "-o",
                mprage_mni_output,
            ]
            LOG.debug(applytransforms_cmd)
            self.runcmd(
                applytransforms_cmd,
                struc_orig.fname.replace(".nii.gz", "_applytransforms.log"),
                raise_on_error=True,
            )


class CsiAlign(Module):
    """
    Align CSI spectroscopy data to structural images and MNI space

    Each CSI image is associated with the closest preceding MPRAGE structural image
    based on series number
    """

    def __init__(self):
        Module.__init__(self, "csi_align")

    def process(self):
        csi_imgs = self.inimgs("csi", "csi*.nii.gz")
        if not csi_imgs:
            self.no_data("No CSI images found")

        struc_imgs = self.inimgs("mprage", "*mprage*.nii.gz")
        if not struc_imgs:
            self.no_data("No MPRAGE images found")

        fsldir = os.environ.get("FSLDIR", None)
        if not fsldir:
            raise RuntimeError("FSLDIR environment variable not set")

        for csi_img in csi_imgs:
            LOG.info(
                f" - Found CSI image: {csi_img.fname}, series number: {csi_img.seriesnumber}"
            )
            matched_struc_img = None
            for struc_img in struc_imgs:
                if struc_img.seriesnumber < csi_img.seriesnumber:
                    if (
                        matched_struc_img is None
                        or matched_struc_img.seriesnumber < struc_img.seriesnumber
                    ):
                        matched_struc_img = struc_img
            if not matched_struc_img:
                self.bad_data("No matching structural image found")
            LOG.info(
                f"   - Matched structural image: {matched_struc_img.fname} (series number: {matched_struc_img.seriesnumber})"
            )

            LOG.info("   - Resampling spectroscopy data to MPRAGE")
            voxel_data = csi_img.data
            # Resample each spectral sample (4th dimension) separately
            resampled_samples = []
            for sample_idx in range(voxel_data.shape[3]):
                # Extract 3D volume for this spectral sample
                sample_3d = voxel_data[:, :, :, sample_idx]
                sample_nii = nib.Nifti1Image(sample_3d, csi_img.affine)

                # Resample this 3D volume to MPRAGE space
                resampled_sample = processing.resample_from_to(
                    sample_nii, matched_struc_img.nii
                )
                resampled_samples.append(resampled_sample.get_fdata())

            # Stack all resampled samples back into 4D
            resampled_4d = np.stack(resampled_samples, axis=3)
            LOG.info(f"   - Resampled shape: {resampled_4d.shape}")
            resampled_nii = nib.Nifti1Image(resampled_4d, matched_struc_img.affine)
            resampled_nii.to_filename(
                self.outfile(csi_img.fname.replace(".nii.gz", "_struc.nii.gz"))
            )

            meandata = np.mean(resampled_4d, axis=3)
            meandata_nii = nib.Nifti1Image(meandata, matched_struc_img.affine)
            meandata_fpath = self.outfile(
                csi_img.fname.replace(".nii.gz", "_mean_struc.nii.gz")
            )
            meandata_nii.to_filename(meandata_fpath)

            # Apply ANTs transforms to spectroscopy data
            ants_warp = matched_struc_img.fname.replace(
                ".nii.gz", "_brain_1Warp.nii.gz"
            )
            ants_affine = matched_struc_img.fname.replace(
                ".nii.gz", "_brain_0GenericAffine.mat"
            )
            ants_warp = os.path.join(self.outfile("../struc_reg"), ants_warp)
            ants_affine = os.path.join(self.outfile("../struc_reg"), ants_affine)
            if not os.path.exists(ants_warp) or not os.path.exists(ants_affine):
                self.bad_data(
                    f"Could not find ANTs transforms: {ants_warp}, {ants_affine} - will not transform to MNI space"
                )

            LOG.info("   - Applying ANTs MNI space transforms to spectroscopy data")
            mni_mean_fname = self.outfile(
                csi_img.fname.replace(".nii.gz", "_mean_mni.nii.gz")
            )
            applytransforms_mean_cmd = [
                "antsApplyTransforms",
                "-d",
                "3",
                "-i",
                meandata_fpath,
                "-r",
                f"{fsldir}/data/standard/MNI152_T1_2mm_brain.nii.gz",
                "-t",
                ants_warp,
                "-t",
                ants_affine,
                "-o",
                mni_mean_fname,
            ]
            LOG.debug(applytransforms_mean_cmd)
            self.runcmd(
                applytransforms_mean_cmd,
                csi_img.fname.replace(".nii.gz", "_applytransforms.log"),
                raise_on_error=True,
            )

            mni_vols = []
            for sample_idx in range(resampled_4d.shape[3]):
                vol = resampled_4d[:, :, :, sample_idx]
                vol_nii = nib.Nifti1Image(vol, matched_struc_img.affine)
                temp_input = self.outfile(f"temp_{sample_idx}.nii.gz")
                temp_output = self.outfile(f"temp_{sample_idx}_mni.nii.gz")
                vol_nii.to_filename(temp_input)
                applytransforms_cmd = [
                    "antsApplyTransforms",
                    "-d",
                    "3",
                    "-i",
                    temp_input,
                    "-r",
                    f"{fsldir}/data/standard/MNI152_T1_2mm_brain.nii.gz",
                    "-t",
                    ants_warp,
                    "-t",
                    ants_affine,
                    "-o",
                    temp_output,
                ]
                LOG.debug(applytransforms_cmd)
                self.runcmd(
                    applytransforms_cmd, self.outfile("temp.log"), raise_on_error=True
                )
                mni_nii = nib.load(temp_output)
                mni_vols.append(mni_nii.get_fdata())
                os.system(f"rm {temp_input} {temp_output} {self.outfile('temp.log')}")

            mni_vols_4d = np.stack(mni_vols, axis=3)
            csi_mni_output = self.outfile(
                csi_img.fname.replace(".nii.gz", "_mni.nii.gz")
            )
            nib.save(nib.Nifti1Image(mni_vols_4d, mni_nii.affine), csi_mni_output)
            LOG.info(f"   - CSI in MNI space: {csi_mni_output}")


MODULES = [
    StrucPreproc(),
    StrucReg(),
    CsiAlign(),
]
