"""
DIFFAD: Processing pipeline for diffusion MRI Alzheimer's project
"""
# TODO packaging and documentation
# summary of how to run on cluster etc
# TODO whole brain tract seg see links in SW Teams chat
# TODO add fsl_streamlines
# TODO centroid of streamlines (weighted by density?)

import logging
import os
import shutil

import numpy as np
import ants

from dipy.core.gradients import gradient_table
from dipy.denoise.localpca import mppca
from dipy.denoise.gibbs import gibbs_removal
from dipy.reconst.dti import (
    TensorModel,
    lower_triangular,
    fractional_anisotropy,
)
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel, auto_response_ssst, response_from_mask_ssst)
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion, ActStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.tracker import probabilistic_tracking
from dipy.tracking.utils import seeds_from_mask
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram

from nifreeze.data import dmri
from nifreeze.estimator import Estimator

from fsort.image_file import ImageFile
from fproc.module import Module

LOG = logging.getLogger(__name__)

WM = [
    2,
    7,
    28,
    30,
    31,
    78,
    100,
    41,
    60,
    62,
    63,
    79,
    108,
    109,
    117,
    77,
    85,
    192,
    251,
    252,
    253,
    254,
    255,
    46,
    16,
]
GM = [
    3,
    8,
    10,
    11,
    12,
    13,
    17,
    18,
    26,
    96,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    42,
    49,
    50,
    51,
    52,
    53,
    54,
    58,
    97,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    80,
    47,
]
CSF = [4, 5, 14, 15, 43, 44, 24, 72]

SYNTHSEG_REGIONS = {
    0: "Background",
    2: "L_Cerebral WM",
    3: "L_Cerebral GM",
    4: "L_Ventricle",
    5: "L_Ventricle",
    7: "L_Cerebellum WM",
    8: "L_Cerebellum GM",
    10: "L_Thal",
    11: "L_Caud",
    12: "L_Puta",
    13: "L_Pall",
    14: "L_3rd ventricle",
    15: "L_4th ventricle",
    16: "Brainstem",
    17: "L_Hipp",
    18: "L_Amyg",
    24: "CSF",
    26: "L_Accu",
    28: "L_ventral DC",
    30: "L_vessel",
    31: "L_choroid plexus",
    78: "L_WM hyper-intensity",
    96: "L_Amygdala-Anterior",
    41: "R_Cerebral WM",
    42: "R_Cerebral GM",
    43: "R_Ventricle",
    44: "R_Ventricle",
    46: "R_Cerebellum_WM",
    47: "R_Cerebellum_GM",
    49: "R_Thal",
    50: "R_Caud",
    51: "R_Puta",
    52: "R_Pall",
    53: "R_Hipp",
    54: "R_Amyg",
    58: "R_Accu",
    60: "R_ventral DC",
    62: "R_vessel",
    63: "R_choroid plexus",
    72: "5th ventricle",
    77: "WM hypo-intensity",
    79: "R_WM hyper-intensity",
    80: "non-WM hypo-intensity",
    85: "Optic chiasm",
    97: "R_Amygdala-Anterior",
    100: "L_wm-intensity-abnormality",
    101: "L_caudate-intensity-abnormality",
    102: "L_putamen-intensity-abnormality",
    103: "L_accumbens-intensity-abnormality",
    104: "L_pallidum-intensity-abnormality",
    105: "L_amygdala-intensity-abnormality",
    106: "L_hippocampus-intensity-abnormality",
    107: "L_thalamus-intensity-abnormality",
    108: "L_VDC-intensity-abnormality",
    109: "R_wm-intensity-abnormality",
    110: "R_caudate-intensity-abnormality",
    111: "R_putamen-intensity-abnormality",
    112: "R_accumbens-intensity-abnormality",
    113: "R_pallidum-intensity-abnormality",
    114: "R_amygdala-intensity-abnormality",
    115: "R_hippocampus-intensity-",
    116: "R_thalamus-intensity-abnormality",
    117: "R_VDC-intensity-abnormality",
    192: "CC",
    251: "CC",
    252: "CC",
    253: "CC",
    254: "CC",
    255: "CC",
}

class BrcPipeline(Module):
    """
    Runs the BRC Pipeline
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "brc", **kwargs)

    def process(self):
        t1 = self.single_inimg("mprage", "mprage.nii.gz")
        t2 = self.single_inimg("flair", "flair.nii.gz")
        if t1 is None:
            self.no_data("No T1 data found in mprage")
        if t2 is None:
            self.no_data("No T2 data found in flair")

        LOG.info("Running BRC pipeline for structural data")
        self.runcmd(
            [
                "struc_preproc.sh",
                "--input",
                t1.fpath,
                "--t2",
                t2.fpath,
                "--path",
                self.outdir,
                "--subject",
                "",
                "--qc",
            ],
            logfile="brc_struc.log",
        )


class BIDSDir(Module):
    """
    Put the data into BIDs format for e.g. MRIQC
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "bids", **kwargs)

    def process(self):
        os.makedirs(self.outfile("sub-01/ses-01/anat"), exist_ok=True)
        os.makedirs(self.outfile("sub-01/ses-01/dwi"), exist_ok=True)
        t1 = self.single_inimg("mprage", "mprage.nii.gz")
        t1.save(self.outfile("sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"))
        t2 = self.single_inimg("flair", "flair.nii.gz")
        t2.save(self.outfile("sub-01/ses-01/anat/sub-01_ses-01_T2w.nii.gz"))

        dti_pa = self.single_inimg("dti", "dti_pa.nii.gz")
        if dti_pa is not None:
            dti_pa.save(self.outfile("sub-01/ses-01/dwi/sub-01_ses-01_dir-pa_dwi.nii.gz"))
        dti_ap = self.single_inimg("dti", "dti_ap.nii.gz", warn=False)
        if dti_ap is not None:
            dti_ap.save(self.outfile("sub-01/ses-01/dwi/sub-01_ses-01_dir-ap_dwi.nii.gz"))
        dti_lr = self.single_inimg("dti", "dti_lr.nii.gz", warn=False)
        if dti_lr is not None:
            dti_lr.save(self.outfile("sub-01/ses-01/dwi/sub-01_ses-01_dir-lr_dwi.nii.gz"))
        dti_rl = self.single_inimg("dti", "dti_rl.nii.gz", warn=False)
        if dti_rl is not None:
            dti_rl.save(self.outfile("sub-01/ses-01/dwi/sub-01_ses-01_dir-rl_dwi.nii.gz"))

        with open(self.outfile("dataset_description.json"), "w") as f:
            f.write('{"Name" : "diffad dataset", "BIDSVersion" : "1.0.2"}')
        with open(self.outfile("participants.tsv"), "w") as f:
            f.write("participant_id\n")
            f.write("sub-01\n")


class MRIQC(Module):
    """
    Compute MRI Quality Control metrics
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "mriqc", deps=["bids"], **kwargs)

    def process(self):
        LOG.info("Running MRIQC")
        retval = self.runcmd(
            [
                "singularity",
                "run",
                "--nv",
                "/gpfs01/software/imaging/mriqc/mriqc-24.0.2.sif",
                self.outfile("../bids"),
                self.outfile(""),
                "participant",
                "--participant-label",
                "sub-01"
            ],
            logfile="mriqc.log",
        )
        if retval != 0:
            self.bad_data(f"MRIQC failed with return code {retval}")


class StrucPreproc(Module):
    """
    BRC Pipeline-like Structural Preprocessing but without FSL
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "struc", **kwargs)
        self.spaces = {"t2_orig": {}, "t1_orig": {}, "t1": {}, "std": {}}
        self.reg = {}

    def _save(self, img, name, space):
        fname = self.outfile(os.path.join(space, f"{name}.nii.gz"))
        img.save(fname)
        self.spaces[space][name] = img
        self.spaces[space][f"{name}_ants"] = ants.image_read(fname)

    def _save_ants(self, img_ants, name, space):
        fname = self.outfile(os.path.join(space, f"{name}.nii.gz"))
        img_ants.image_write(fname)
        self.spaces[space][name] = ImageFile(fname, warn_json=False)
        self.spaces[space][f"{name}_ants"] = img_ants

    def _fromfile(self, name, space):
        fname = self.outfile(os.path.join(space, f"{name}.nii.gz"))
        img = ImageFile(fname, warn_json=False)
        self.spaces[space][name] = img
        self.spaces[space][f"{name}_ants"] = ants.image_read(fname)

    def _fname(self, name, space):
        return self.spaces[space][name].fpath

    def _brain_extract(self, name, space):
        self.runcmd(
            [
                "mri_synthstrip",
                "-i",
                self._fname(name, space),
                "-o",
                self.outfile(f"{space}/{name}_brain.nii.gz"),
                "-m",
                self.outfile(f"{space}/{name}_brain_mask.nii.gz"),
            ],
            logfile="mri_synthstrip_t1.log",
        )
        self._fromfile(f"{name}_brain", space)
        self._fromfile(f"{name}_brain_mask", space)

    def _get_stdref(self):
        fsl_stdref_path = (
            f"{os.environ.get('FSLDIR', '')}/data/standard/MNI152_T1_1mm.nii.gz"
        )
        stdref_path = self.kwargs.get("stdref", fsl_stdref_path)
        LOG.info(" - Using standard space reference: %s", stdref_path)
        stdref = ImageFile(stdref_path, warn_json=False)
        self._save(stdref, "ref", "std")
        fsl_stdref_brain_mask_path = f"{os.environ.get('FSLDIR', '')}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
        stdref_brain_mask_path = self.kwargs.get(
            "stdref_brain", fsl_stdref_brain_mask_path
        )
        LOG.info(" - Using standard space reference brain: %s", stdref_brain_mask_path)
        stdref_brain_mask = ImageFile(stdref_brain_mask_path, warn_json=False)
        self._save(stdref_brain_mask, "stdref_brain_mask", "std")
        fov_mask = np.ones_like(stdref.data, dtype=np.uint8)
        stdref.save_derived(fov_mask, self.outfile("std/fov_mask.nii.gz"))
        self._fromfile("fov_mask", "std")
        brc_deface_path = "/software/imaging/BRC_pipeline/1.7.1/global/templates/MNI152_T1_1mm_BigFoV_facemask.nii.gz"
        deface_mask_path = self.kwargs.get("deface_mask_std", brc_deface_path)
        LOG.info(" - Using deface mask: %s", deface_mask_path)
        deface_mask_ants = ants.image_read(deface_mask_path)
        self._save_ants(deface_mask_ants, "deface_mask_full", "std")
        deface_mask_std_ants = ants.resample_image_to_target(
            deface_mask_ants, self.spaces["std"]["ref_ants"], interp_type="genericLabel"
        )
        self._save_ants(deface_mask_std_ants, "deface_mask", "std")

    def _reg(
        self,
        name,
        from_space,
        to_space,
        type_of_transform="Affine",
        initial_transform=None,
    ):
        img_ants = self.spaces[from_space][f"{name}_ants"]
        ref_ants = self.spaces[to_space]["ref_ants"]
        reg_result = ants.registration(
            ref_ants,
            img_ants,
            initial_transform=initial_transform,
            type_of_transform=type_of_transform,
        )

        self._save_ants(reg_result["warpedmovout"], name, to_space)
        if from_space not in self.reg:
            self.reg[from_space] = {}
        if to_space not in self.reg:
            self.reg[to_space] = {}
        # self.reg[from_space][to_space] = reg_result["fwdtransforms"]
        # self.reg[to_space][from_space] = reg_result["invtransforms"]

        if type_of_transform == "SyN":
            shutil.copyfile(
                reg_result["fwdtransforms"][0],
                self.outfile(f"reg/{from_space}_to_{to_space}_nonlin.nii.gz"),
            )
            shutil.copyfile(
                reg_result["fwdtransforms"][1],
                self.outfile(f"reg/{from_space}_to_{to_space}_lin.mat"),
            )
            shutil.copyfile(
                reg_result["invtransforms"][1],
                self.outfile(f"reg/{to_space}_to_{from_space}_nonlin.nii.gz"),
            )
            shutil.copyfile(
                reg_result["invtransforms"][0],
                self.outfile(f"reg/{to_space}_to_{from_space}_lin.mat"),
            )
            self.reg[from_space][to_space] = [
                self.outfile(f"reg/{from_space}_to_{to_space}_nonlin.nii.gz"),
                self.outfile(f"reg/{from_space}_to_{to_space}_lin.mat"),
            ]
            self.reg[to_space][from_space] = [
                self.outfile(f"reg/{to_space}_to_{from_space}_lin.mat"),
                self.outfile(f"reg/{to_space}_to_{from_space}_nonlin.nii.gz"),
            ]
        else:
            shutil.copyfile(
                reg_result["fwdtransforms"][0],
                self.outfile(f"reg/{from_space}_to_{to_space}_lin.mat"),
            )
            shutil.copyfile(
                reg_result["invtransforms"][0],
                self.outfile(f"reg/{to_space}_to_{from_space}_lin.mat"),
            )
            self.reg[from_space][to_space] = [
                self.outfile(f"reg/{from_space}_to_{to_space}_lin.mat")
            ]
            self.reg[to_space][from_space] = [
                self.outfile(f"reg/{to_space}_to_{from_space}_lin.mat")
            ]

    def _transform(self, name, from_space, to_space, is_roi=False):
        src = self.spaces[from_space][f"{name}_ants"]
        dest_ref = self.spaces[to_space]["ref_ants"]
        transform = self.reg[from_space][to_space]
        if transform == "identity":
            dest = ants.resample_image_to_target(
                src, dest_ref, interp_type="genericLabel" if is_roi else "linear"
            )
        else:
            dest = ants.apply_transforms(
                dest_ref,
                src,
                transform,
                interpolator="genericLabel" if is_roi else "linear",
            )
        self._save_ants(dest, name, to_space)

    def _crop(self, name, space, fov_space, to_space):
        if "fov_mask" not in self.spaces[space]:
            self._transform("fov_mask", fov_space, space, is_roi=True)
        nonzero = np.nonzero(self.spaces[space]["fov_mask"].data)
        bbox = [slice(np.min(nonzero[i]), np.max(nonzero[i]) + 1) for i in range(3)]
        LOG.info(f" - Cropping {name} image using bounding box: {bbox}")
        img = self.spaces[space][name]
        cropped_data = img.data[tuple(bbox)]
        cropped_affine = np.array(img.affine)
        cropped_affine[:3, 3] = img.affine[:3, 3] + np.dot(
            img.affine[:3, :3], [bbox[i].start for i in range(3)]
        )
        import nibabel as nib

        cropped_nii = nib.Nifti1Image(
            cropped_data, affine=cropped_affine, header=img.header
        )
        cropped_nii.to_filename(self.outfile(f"{to_space}/{name}.nii.gz"))
        self._fromfile(name, to_space)
        if "ref" not in self.spaces[to_space]:
            for s in self.reg:
                if space in self.reg[s]:
                    self.reg[s][to_space] = self.reg[s][space]
            self.reg[to_space] = dict(self.reg[space])
            self.reg[space][to_space] = "identity"
            self.reg[to_space][space] = "identity"
            self._save(self.spaces[to_space][name], "ref", to_space)

    def _deface(self, name, space):
        head_img = self.spaces[space][name]
        if "deface_mask" not in self.spaces[space]:
            self._transform("deface_mask_full", "std", space, is_roi=True)
            self._save(self.spaces[space]["deface_mask_full"], "deface_mask", space)
        deface_mask = self.spaces[space]["deface_mask"]
        defaced_data = head_img.data * (deface_mask.data == 0).astype(np.uint8)
        head_img.save_derived(
            defaced_data, self.outfile(f"{space}/{name}_defaced.nii.gz")
        )
        self._fromfile(f"{name}_defaced", space)

    def _biasfield(self, name, mask_name, space):
        biasfield = ants.n4_bias_field_correction(
            self.spaces[space][f"{name}_ants"],
            self.spaces[space][f"{mask_name}_ants"],
            return_bias_field=True,
            shrink_factor=4,
        )
        biasfield = biasfield.numpy()
        biasfield = biasfield / np.mean(biasfield)
        biasfield[np.isclose(biasfield, 0, atol=1e-3)] = 1  # Avoid division by zero
        biasfield[~np.isfinite(biasfield)] = 1  # Remove any non-finite values
        self.spaces[space][name].save_derived(
            biasfield, self.outfile(f"{space}/biasfield.nii.gz")
        )
        self._fromfile("biasfield", space)

    def _biascorr(self, name, space):
        img = self.spaces[space][name]
        biasfield = self.spaces[space]["biasfield"].data

        # May have invalid bias values due to different FOVs
        biasfield[np.isclose(biasfield, 0, atol=1e-3)] = 1
        biasfield[~np.isfinite(biasfield)] = 1
        data_biascorr = img.data / biasfield
        img.save_derived(data_biascorr, self.outfile(f"{space}/{name}_biascorr.nii.gz"))
        self._fromfile(f"{name}_biascorr", space)

    def _seg(self, name, space):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.runcmd(
            [
                "mri_synthseg",
                "--i",
                self.outfile(f"{space}/{name}.nii.gz"),
                "--o",
                self.outfile(f"seg/{name}_seg.nii.gz"),
                "--vol",
                self.outfile(f"seg/{name}_seg.csv"),
                "--cpu",
                "--robust",
            ],
            logfile="mri_synthseg.log",
        )

        seg = ImageFile(self.outfile(f"seg/{name}_seg.nii.gz"), warn_json=False)
        wm = np.zeros_like(seg.data, dtype=np.uint8)
        gm = np.zeros_like(seg.data, dtype=np.uint8)
        csf = np.zeros_like(seg.data, dtype=np.uint8)
        for region in SYNTHSEG_REGIONS:
            if region in GM:
                gm[seg.data == region] = 1
            elif region in WM:
                wm[seg.data == region] = 1
            elif region in CSF:
                csf[seg.data == region] = 1
        seg.save_derived(wm, self.outfile(f"seg/{name}_wm.nii.gz"))
        seg.save_derived(gm, self.outfile(f"seg/{name}_gm.nii.gz"))
        seg.save_derived(csf, self.outfile(f"seg/{name}_csf.nii.gz"))

    def process(self):
        for space in self.spaces:
            os.makedirs(self.outfile(space), exist_ok=True)
        os.makedirs(self.outfile("seg"), exist_ok=True)
        os.makedirs(self.outfile("reg"), exist_ok=True)

        t1 = self.single_inimg("mprage", "mprage.nii.gz")
        t2 = self.single_inimg("flair", "flair.nii.gz")
        if t1 is None:
            self.no_data("No T1 data found in mprage")
        if t2 is None:
            self.no_data("No T2 data found in flair")

        LOG.info(" - Reorienting T1 and T2 images to standard orientation")
        t1_orig = t1.reorient2std()
        self._save(t1_orig, "t1", "t1_orig")
        self._save(t1_orig, "ref", "t1_orig")
        t2_orig = t2.reorient2std()
        self._save(t2_orig, "t2", "t2_orig")
        self._save(t2_orig, "ref", "t2_orig")

        LOG.info(" - Doing brain extraction of T1 image using mri_synthstrip")
        self._brain_extract("t1", "t1_orig")

        LOG.info(" - Performing linear registration of T1 to standard space reference")
        self._get_stdref()
        self._reg("t1", "t1_orig", "std", type_of_transform="Affine")

        LOG.info(
            " - Performing nonlinear registration of T1 to standard space reference"
        )
        self._reg(
            "t1",
            "t1_orig",
            "std",
            type_of_transform="SyN",
            initial_transform=self.reg["t1_orig"]["std"],
        )

        LOG.info(" - Converting standard space brain mask to T1 space")
        self._transform("stdref_brain_mask", "std", "t1_orig", is_roi=True)

        LOG.info(" - Cropping T1 to same FOV as standard space reference")
        self._crop("t1", "t1_orig", "std", "t1")

        LOG.info(" - Defacing T1 image")
        self._deface("t1", "t1")
        self._deface("t1", "std")

        LOG.info(" - Performing bias field correction on T1 image")
        self._biasfield("t1", "stdref_brain_mask", "t1_orig")
        self._transform("biasfield", "t1_orig", "t1")
        self._transform("biasfield", "t1_orig", "std")
        self._biascorr("t1", "t1_orig")
        self._biascorr("t1", "t1")
        self._biascorr("t1", "std")
        self._biascorr("t1_defaced", "t1")
        self._biascorr("t1_defaced", "std")

        LOG.info(" - Processing T2 image")
        self._brain_extract("t2", "t2_orig")

        LOG.info(" - Performing linear registration of T2 to T1")
        self._reg("t2", "t2_orig", "t1_orig", type_of_transform="Rigid")
        self.reg["t2_orig"]["std"] = (
            self.reg["t2_orig"]["t1_orig"] + self.reg["t1_orig"]["std"]
        )
        self.reg["std"]["t2_orig"] = (
            self.reg["std"]["t1_orig"] + self.reg["t1_orig"]["t2_orig"]
        )

        LOG.info(" - Transforming T2 brain to T1 space")
        self._transform("t2_brain", "t2_orig", "t1_orig")
        self._transform("t2_brain_mask", "t2_orig", "t1_orig", is_roi=True)

        LOG.info(" - Cropping T2 images to same FOV as T1")
        self.reg["t2_orig"]["t1"] = self.reg["t2_orig"]["t1_orig"]
        self.reg["t1"]["t2_orig"] = self.reg["t1_orig"]["t2_orig"]
        self._transform("t2", "t2_orig", "t1")
        self._transform("t2_brain", "t2_orig", "t1")
        self._transform("t2_brain_mask", "t2_orig", "t1", is_roi=True)

        LOG.info(" - Transforming T2 images to std space")
        self._transform("t2", "t2_orig", "std")
        self._transform("t2_brain", "t2_orig", "std")
        self._transform("t2_brain_mask", "t2_orig", "std", is_roi=True)

        LOG.info(" - Defacing T2 image")
        self._deface("t2", "t1")
        self._deface("t2", "std")

        LOG.info(" - Performing bias field correction on T2 image")
        self._transform("biasfield", "t1_orig", "t2_orig")
        self._biascorr("t2", "t2_orig")
        self._biascorr("t2", "t1_orig")
        self._biascorr("t2", "t1")
        self._biascorr("t2", "std")
        self._biascorr("t2_defaced", "t1")
        self._biascorr("t2_defaced", "std")

        LOG.info(" - Performing tissue segmentation on bias-corrected T1 image")
        self._seg("t1_biascorr", "t1")

        # TODO freesurfer?


class DtiPreproc(Module):
    """
    Identify DTI data
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "dtipreproc", **kwargs)

    def process(self):
        src = self.kwargs.get("src", "dti")
        glob = self.kwargs.get("glob", "dti*.nii.gz")
        dtis = self.inimgs(src, glob)
        if not dtis:
            self.no_data(f"No DTI data found in {src}/{glob}")
        
        dti_files = {}
        for pedir in ("ap", "pa", "lr", "rl"):
            files = [f for f in dtis if f.fname_noext.endswith(f"_{pedir}")]
            if len(files) > 1:
                LOG.warning(f"Found multiple DTI files for {pedir}: {files} - using first")
            dti_files[pedir] = None if not files else files[0]

        if dti_files["ap"] and dti_files["pa"]:
            up, down = "pa", "ap"
            LOG.info(f"Using PA as up and AP as down for distortion correction")
            if dti_files["lr"] or dti_files["rl"]:
                LOG.warn(f"Also found LR/RL data but will not use it")
        elif dti_files["lr"] and dti_files["rl"]:
            up, down = "lr", "rl"
            LOG.info(f"Using LR as up and RL as down for distortion correction")
            if dti_files["ap"] or dti_files["pa"]:
                LOG.warn(f"Also found AP/PA data but will not use it")
        elif dti_files["pa"] or dti_files["ap"]:
            up, down = "pa" if dti_files["pa"] else "ap", None
            LOG.info(f"Using {'PA' if dti_files['pa'] else 'AP'} data only")
            if dti_files["lr"] or dti_files["rl"]:
                LOG.warn(f"Also found LR/RL data but will not use it")
        elif dti_files["lr"] or dti_files["rl"]:
            up, down = "lr" if dti_files["lr"] else "rl", None
            LOG.info(f"Using {'LR' if dti_files['lr'] else 'RL'} data only")
            if dti_files["ap"] or dti_files["pa"]:
                LOG.warn(f"Also found AP/PA data but will not use it")
        else:
            self.no_data("No DTI data found")

        dti_files[up].save(self.outfile("dti_up.nii.gz"))
        if down:
            dti_files[down].save(self.outfile("dti_down.nii.gz"))

        with open(self.outfile("dti_files.csv"), "w") as f:
            f.write(f"up,{up}\n")
            if down:
                f.write(f"down,{down}\n")

class DtiDenoise(Module):
    """
    Denoising of diffusion data
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "denoise", deps=["dtipreproc"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "dtipreproc")
        glob = self.kwargs.get("glob", "dti*.nii.gz")
        dtis = self.inimgs(src, glob)
        if not dtis:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        for dti in dtis:
            LOG.info(f"Doing MPPCA denoising of: {dti.fname}")
            denoised_data = mppca(dti.data, patch_radius=10)
            dti.save_derived(
                denoised_data, self.outfile(f"{dti.fname_noext}_denoised.nii.gz"), copy_bdata=True
            )

            rms_diff = np.sqrt((dti.data - denoised_data) ** 2)
            dti.save_derived(rms_diff, self.outfile(f"{dti.fname_noext}_residuals.nii.gz"))


class DtiUnring(Module):
    """
    Unringing of diffusion data
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "unring", deps=["denoise"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "denoise")
        glob = self.kwargs.get("glob", "dti*_denoised.nii.gz")
        dtis = self.inimgs(src, glob)
        if not dtis:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        for dti in dtis:
            LOG.info(f"Unringing DTI data: {dti.fname}")
            unringed = gibbs_removal(dti.data)
            dti.save_derived(
                unringed, self.outfile(f"{dti.fname_noext}_unringed.nii.gz"), copy_bdata=True
            )


class DtiDistCorr(Module):

    def __init__(self, **kwargs):
        Module.__init__(self, "distcorr", deps=["unring"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "unring")
        glob_up = self.kwargs.get("glob", "dti_up*.nii.gz")
        dti_up = self.single_inimg(src, glob_up)
        glob_down = self.kwargs.get("glob", "dti_down*.nii.gz")
        dti_down = self.single_inimg(src, glob_down)
        if dti_up is None:
            self.no_data(f"No DTI data found in {src}/{glob_up}")

        if dti_down is None:
            LOG.warn(f"No DTI down data found in {src}/{glob_down} - distcorr will not be performed")
            dti_up.save(self.outfile("dti_up_distcorr.nii.gz"))
            dti_up.save(self.outfile("dti_distcorr.nii.gz"))
            return

        # Copy metadata for distortion corrected output
        for ext in ("json", "bval", "bvec"):
            shutil.copyfile(dti_up.fpath.replace(".nii.gz", f".{ext}"), self.outfile("dti_up_distcorr.json"))
            shutil.copyfile(dti_down.fpath.replace(".nii.gz", f".{ext}"), self.outfile("dti_down_distcorr.json"))

        bval_up, bvec_up = dti_up.bval, dti_up.bvec
        bval_down, bvec_down = dti_down.bval, dti_down.bvec
        bval = np.concatenate([bval_up, bval_down], axis=0)
        bvec = np.concatenate([bvec_up, bvec_down], axis=1)
        shutil.copyfile(dti_up.fpath.replace(".nii.gz", f".json"), self.outfile("dti_distcorr.json"))
        np.savetxt(self.outfile("dti_distcorr.bval"), bval, fmt="%.3f")
        np.savetxt(self.outfile("dti_distcorr.bvec"), bvec, fmt="%.6f")

        #t2 = self.single_inimg("flair", "flair.nii.gz")
        #if t2 is None:
        #    self.no_data("No T2 data found in flair")

        LOG.info("Running TORTOISEProcess_cuda for distortion correction")
        retval = self.runcmd(
            [
                "TORTOISEProcess_cuda",
                "--up_data",
                dti_up.fpath,
                "--up_json",
                dti_up.json_fpath,
                "--ub",
                dti_up.bval_fpath,
                "--uv",
                dti_up.bvec_fpath,
                "--down_data",
                dti_down.fpath,
                "--db",
                dti_down.bval_fpath,
                "--dv",
                dti_down.bvec_fpath,
                "--denoising",
                "off",
                "--gibbs",
                "",
                "-c",
                "off",
                "--s2v",
                "0",
                "--repol",
                "0",
                #"-s",
                #t2.fpath,
                "-o",
                self.outfile("tortoise_output.nii"),
                "-t",
                self.outfile("temp"),
            ],
            logfile="tortoise.log",
        )
        if retval != 0:
            self.bad_data(f"TORTOISEProcess_cuda failed with return code {retval}")

        corrected_up = self.single_inimg(self.name, f"temp/{dti_up.fname_noext}_proc_final_temp.nii*")
        corrected_down = self.single_inimg(self.name, f"temp/{dti_down.fname_noext}_proc_final_temp.nii*")
        corrected_up.save(self.outfile("dti_up_distcorr.nii.gz"))
        corrected_down.save(self.outfile("dti_down_distcorr.nii.gz"))
        combined = ImageFile(self.outfile("tortoise_output.nii"), warn_json=False)
        combined.save(self.outfile("dti_distcorr.nii.gz"))

class DtiBrainMask(Module):
    """
    Extract mean B0 and perform brain extraction using mri_synthstrip
    """
    def __init__(self, **kwargs):
        Module.__init__(self, "brainmask", deps=["distcorr"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "distcorr")
        glob = self.kwargs.get("glob", "dti_distcorr.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        # Extract mean B0
        b0s = dti.bval < self.pipeline.options.b0_threshold
        if not np.any(b0s):
            self.no_data(f"No B0 volumes found in the DTI data (bval < {self.pipeline.options.b0_threshold})")
        b0_volumes = dti.data[..., b0s]
        dti.save_derived(b0_volumes, self.outfile("b0.nii.gz"))
        mean_b0 = np.mean(b0_volumes, axis=-1)
        mean_b0_path = self.outfile("mean_b0.nii.gz")
        dti.save_derived(mean_b0, mean_b0_path)

        # Brain extraction using mri_synthstrip
        LOG.info("Running mri_synthstrip for brain extraction on mean B0")
        self.runcmd([
            "mri_synthstrip",
            "-i", mean_b0_path,
            "-o", self.outfile("b0_brain.nii.gz"),
            "-m", self.outfile("b0_brain_mask.nii.gz"),
        ], logfile="mri_synthstrip_b0.log")

class DtiReg(Module):
    """
    Register DTI to structural data
    """
    def __init__(self, **kwargs):
        Module.__init__(self, "dtireg", deps=["brainmask", "distcorr", "struc"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "distcorr")
        glob = self.kwargs.get("glob", "dti_distcorr.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        dti_brain = self.single_inimg("brainmask", "b0_brain.nii.gz")
        if dti_brain is None:
            self.no_data("No DTI brain image found in brainmask/b0_brain.nii.gz")

        t1_src = self.kwargs.get("t1_src", "struc/t1_orig")
        t1_glob = self.kwargs.get("t1_glob", "t1_brain.nii.gz")
        t1_brain = self.single_inimg(t1_src, t1_glob)
        if t1_brain is None:
            self.no_data(f"No T1 brain image found in {t1_src}/{t1_glob}")

        LOG.info("Registering DTI to T1 image")
        reg_result = ants.registration(
            ants.image_read(t1_brain.fpath),
            ants.image_read(dti_brain.fpath),
            type_of_transform="Affine",
        )
        dti_reg = reg_result["warpedmovout"]
        ants.image_write(dti_reg, self.outfile("dti_brain_t1.nii.gz"))

        dti_brain_t1 = ants.apply_transforms(
            ants.image_read(dti_brain.fpath),
            ants.image_read(t1_brain.fpath),
            reg_result["fwdtransforms"],
            interpolator="linear",
            whichtoinvert=[True],
        )
        ants.image_write(dti_brain_t1, self.outfile("t1_brain_dti.nii.gz"))

        # Only save fwd transform because inverse is calculated on the fly
        # using whichtoinvert option
        shutil.copyfile(
            reg_result["fwdtransforms"][0],
            self.outfile(f"dti_to_t1_lin.mat"),
        )

class DtiEddyCorrection(Module):
    """
    Eddy current motion correction
    """
    def __init__(self, **kwargs):
        Module.__init__(self, "eddycorr", deps=["brainmask", "distcorr"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "distcorr")
        glob = self.kwargs.get("glob", "dti_distcorr.nii.gz")
        dti = self.single_inimg(src, glob)
        if not dti:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        mean_b0 = self.single_inimg("brainmask", "mean_b0.nii.gz")
        mask = self.single_inimg("brainmask", "b0_brain_mask.nii.gz")

        eddycorr_info = dmri.from_nii(
            dti.fpath,
            bvec_file=dti.fpath.replace(".nii.gz", ".bvec"),
            bval_file=dti.fpath.replace(".nii.gz", ".bval"),
            b0_file=mean_b0.fpath,
            brainmask_file=mask.fpath,
        )
        estimator = Estimator("dti")
        estimated_affine = estimator.run(
            eddycorr_info,
            omp_nthreads=1,
            n_jobs=1,
            seed=42,
        )
        eddycorr_info.to_nifti(self.outfile("dti_eddycorr_nob0.nii.gz"))

        # nifreeze only seems to return non-B0 volumes?
        dwi_data_nob0 = ImageFile(self.outfile("../eddycorr/dti_eddycorr_nob0.nii.gz"))
        dwi_data_b0 = self.single_inimg("brainmask", "b0.nii.gz")
        dti_eddycorr_data = np.zeros_like(dti.data)
        b0s = dti.bval < self.pipeline.options.b0_threshold
        dti_eddycorr_data[..., b0s] = dwi_data_b0.data
        dti_eddycorr_data[..., ~b0s] = dwi_data_nob0.data
        dti.save_derived(dti_eddycorr_data, self.outfile("dti_eddycorr.nii.gz"), copy_bdata=True)


class DtiFitting(Module):
    """
    Fitting of diffusion tensor imaging (DTI) data
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "dtifit", deps=["eddycorr"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "eddycorr")
        glob = self.kwargs.get("glob", "dti_eddycorr.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        grad_table = gradient_table(
            np.array(dti.bval),
            np.array(dti.bvec),
            b0_threshold=self.pipeline.options.b0_threshold
        )

        # FIXME mask data
        LOG.info(f"Fitting diffusion tensor model to: {dti.fname}")
        tensor_model = TensorModel(grad_table)
        tensor_fit = tensor_model.fit(dti.data)
        tensor_vals = lower_triangular(tensor_fit.quadratic_form)

        LOG.info("Computing anisotropy measures (FA, MD, RGB)")
        FA = fractional_anisotropy(tensor_fit.evals)
        dti.save_derived(FA, self.outfile("FA.nii.gz"))
        dti.save_derived(tensor_fit.md, self.outfile("MD.nii.gz"))
        for idx in range(3):
            dti.save_derived(
                tensor_fit.evals[..., idx], self.outfile(f"L{idx+1}.nii.gz")
            )
            dti.save_derived(
                tensor_fit.evecs[..., idx], self.outfile(f"V{idx+1}.nii.gz")
            )
        # dti.save_derived(tensor_fit.model_S0, self.outfile("S0.nii.gz"))

class DtiFittingFSL(Module):
    """
    Fitting of diffusion tensor imaging (DTI) data using FSL
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "dtifit_fsl", deps=["eddycorr", "brainmask"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "eddycorr")
        glob = self.kwargs.get("glob", "dti_eddycorr.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        mask = self.single_inimg("brainmask", "b0_brain_mask.nii.gz")
        if mask is None:
            self.no_data("No brain mask found in brainmask/b0_brain_mask.nii.gz")
            
        bval = dti.fpath.replace(".nii.gz", ".bval")
        bvec = dti.fpath.replace(".nii.gz", ".bvec")
    
        self.runcmd([
            "dtifit",
            "-k", dti.fpath,
            "-o", self.outfile(""),
            "-m", mask.fpath,
            "-r", bvec,
            "-b", bval,
        ], logfile="dtifit_fsl.log")
        
class FibreModelling(Module):
    """
    Fitting of fibre orientation distributions (FOD) using constrained spherical deconvolution (CSD)
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "fibremod", deps=["eddycorr", "dtifit", "brainmask"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "eddycorr")
        glob = self.kwargs.get("glob", "dti_eddycorr.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        fa_src = self.kwargs.get("fa_src", "dtifit")
        fa_glob = self.kwargs.get("fa_glob", "FA.nii.gz")
        fa = self.single_inimg(fa_src, fa_glob)
        if fa is None:
            self.no_data(f"No FA data found in {fa_src}/{fa_glob}")

        mask_src = self.kwargs.get("mask_src", "brainmask")
        mask_glob = self.kwargs.get("mask_glob", "b0_brain_mask.nii.gz")
        mask = self.single_inimg(mask_src, mask_glob)
        if mask is None:
            self.no_data(f"No brain mask found in {mask_src}/{mask_glob}")

        gtab = gradient_table(
            np.array(dti.bval),
            np.array(dti.bvec),
            b0_threshold=self.pipeline.options.b0_threshold
        )

        # Processing parameters (tune to your data)
        FA_RESPONSE_THRESH = 0.7 # FA threshold to select single-fiber voxels for response
        FA_SEED_THRESH = 0.2 # FA threshold for seeding mask
        SEED_DENSITY = 2 # seeds per voxel (integer)
        REL_PEAK_THRESH = 0.5 # relative peak threshold for peaks extraction
        MIN_SEP_ANGLE = 25 # min separation angle (degrees)
        STEP_SIZE = 0.5 # mm

        # ---------------------
        # Estimate response function
        # ---------------------
        LOG.info(" - Estimating response function using high-FA voxels...")
        # Approach A: use response_from_mask with a high-FA mask
        masked_data = dti.data * (mask.data[..., None] > 0)
        sf_mask = (fa.data > FA_RESPONSE_THRESH)
        if sf_mask.sum() < 50:
            LOG.warn(f"Too few voxels with FA > {FA_RESPONSE_THRESH}. Falling back to auto_response")
            response, ratio = auto_response_ssst(gtab, masked_data, roi_radius=10, fa_thr=FA_RESPONSE_THRESH)
        else:
            response, ratio = response_from_mask_ssst(gtab, masked_data, sf_mask)

        LOG.info(f" - Response estimated: {response}")

        LOG.info(" - Fitting CSD model")
        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        csd_fit = csd_model.fit(masked_data)

        LOG.info(" - Computing FODs on default sphere")
        fod = csd_fit.odf(default_sphere)
        dti.save_derived(fod, self.outfile("fod.nii.gz"))

        LOG.info(" - Extracting peaks from CSD model (for tractography)")
        peaks = peaks_from_model(
            model=csd_model,
            data=masked_data,
            sphere=default_sphere,
            relative_peak_threshold=REL_PEAK_THRESH,
            min_separation_angle=MIN_SEP_ANGLE,
            mask=mask.data
        )
        #dti.save_derived(peaks, self.outfile("peaks.nii.gz"))

        LOG.info(" - Preparing seeds from FA thresholded mask and stopping criterion...")
        seed_mask = np.logical_and(fa.data > FA_SEED_THRESH, mask.data > 0)
        seeds = utils.seeds_from_mask(seed_mask, density=SEED_DENSITY, affine=dti.affine)
        stopping_criterion = BinaryStoppingCriterion(seed_mask)

        LOG.info(" - Running LocalTracking")
        streamlines_generator = LocalTracking(peaks, stopping_criterion, seeds=seeds, affine=dti.affine, step_size=STEP_SIZE)
        streamlines = list(streamlines_generator)
        LOG.info(f" - Generated {len(streamlines)} streamlines")

        tractogram = StatefulTractogram(streamlines, dti.nii, Space.RASMM)
        out_trk = self.outfile("streamlines.trk")
        LOG.info(f" - Saving streamlines to {out_trk} ...")
        save_trk(tractogram, out_trk)

class FibreModellingFSL(Module):
    """
    Fitting of fibre orientation distributions (FOD) using FSL bedpostx
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "fibremod_fsl", deps=["eddycorr", "brainmask"], **kwargs)

    def process(self):
        src = self.kwargs.get("src", "eddycorr")
        glob = self.kwargs.get("glob", "dti_eddycorr.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        mask_src = self.kwargs.get("mask_src", "brainmask")
        mask_glob = self.kwargs.get("mask_glob", "b0_brain_mask.nii.gz")
        mask = self.single_inimg(mask_src, mask_glob)
        if mask is None:
            self.no_data(f"No brain mask found in {mask_src}/{mask_glob}")

        dti.save(self.outfile("data.nii.gz"))
        mask.save(self.outfile("nodif_brain_mask.nii.gz"))
        shutil.move(dti.fpath.replace(".nii.gz", ".bval"), self.outfile("bvals"))
        shutil.move(dti.fpath.replace(".nii.gz", ".bvec"), self.outfile("bvecs"))

        self.runcmd([
            "bedpostx",
            self.outfile(""),
            "--nf", "3",
            "--fudge", "1",
            "--bi", "3000",
            "--nj", "1250",
            "--se", "25",
            "--model", "2",
            "--cnonlinear",
        ], logfile="bedpostx.log")

class Tractography(Module):
    """
    Anatomical tractography using fibre orientation distributions (FOD)
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "tract", deps=["eddycorr", "fibremod", "struc", "brainmask"], **kwargs)

    def _std_to_dti(self, img_std):
        lin = self.outfile(f"../struc/reg/std_to_t1_orig_lin.mat")
        nonlin = self.outfile(f"../struc/reg/std_to_t1_orig_nonlin.nii.gz")
        dti2struc_lin = self.outfile(f"../dtireg/dti_to_t1_lin.mat")

        strucref_ants = ants.image_read(self.outfile("../struc/t1/t1.nii.gz"))
        img_std_ants = ants.image_read(img_std.fpath)
        img_struc_ants = ants.apply_transforms(
            strucref_ants,
            img_std_ants,
            [lin, nonlin],
            #interpolator="genericLabel",
        )
        ants.image_write(img_struc_ants, self.outfile(f"{img_std.fname_noext}_struc.nii.gz"))
        dtiref_ants = ants.image_read(self.outfile("../brainmask/mean_b0.nii.gz"))
        img_dti_ants = ants.apply_transforms(
            dtiref_ants,
            img_std_ants,
            [lin, nonlin, dti2struc_lin],
            whichtoinvert=[False, False, True]
            #interpolator="genericLabel",
        )
        ants.image_write(img_dti_ants, self.outfile(f"{img_std.fname_noext}_dti2.nii.gz"))
        img_dti = ants.apply_transforms(
            dtiref_ants,
            img_struc_ants,
            [dti2struc_lin],
            whichtoinvert=[True]
            #interpolator="genericLabel",
        )
        dti_outfile = self.outfile(f"{img_std.fname_noext}_dti.nii.gz")
        ants.image_write(img_dti, dti_outfile)
        return ImageFile(dti_outfile, warn_json=False)

    def process(self):
        fsldir = os.getenv("FSLDIR")
        if not fsldir:
            self.bad_data("FSLDIR environment variable not set - FSL is required for tractography")

        src = self.kwargs.get("src", "eddycorr")
        glob = self.kwargs.get("glob", "dti_eddycorr.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src}/{glob}")

        # TODO whole brain tractography -> segment based on XTRACT rois -> streamlines -> centroids
        # Also probtrackx2 -> xtract -> fsl_streamlines -> centroids

        # Proof of concept to do some anatomical tractography using xtract ROIs
        tract = "af_l"
        os.makedirs(self.outfile(tract), exist_ok=True)
        xtract_data = os.path.join(fsldir, "data", "xtract_data", "Human")
        seed = ImageFile(os.path.join(xtract_data, tract, "seed.nii.gz"), warn_json=False)
        target = ImageFile(os.path.join(xtract_data, tract, "target2.nii.gz"), warn_json=False)
        #stop = ImageFile(os.path.join(xtract_data, tract, "stop.nii.gz"), warn_json=False)
        exclude = ImageFile(os.path.join(xtract_data, tract, "exclude.nii.gz"), warn_json=False)
        stdref = ImageFile(os.path.join(fsldir, "data", "standard", "MNI152_T1_1mm_brain.nii.gz"), warn_json=False)
        stdref_dti = self._std_to_dti(stdref)
        
        seed = self._std_to_dti(seed)
        target = self._std_to_dti(target)
        exclude = self._std_to_dti(exclude)
        fod = self.single_inimg("fibremod", "fod.nii.gz")
        seed.save(self.outfile(f"{tract}/seed.nii.gz"))
        target.save(self.outfile(f"{tract}/target.nii.gz"))

        seeds = seeds_from_mask(seed.data, seed.affine, density=10)
        #sc = BinaryStoppingCriterion(target.data)
        sc = ActStoppingCriterion(target.data, exclude.data)

        streamline_generator = probabilistic_tracking(
            seeds,
            sc,
            dti.affine,
            sf=fod.data,
            random_seed=1,
            sphere=default_sphere,
            max_angle=20,
            step_size=0.2,
            return_all=False,
        )
        
        streamlines = list(streamline_generator)
        LOG.info(f" - Generated {len(streamlines)} streamlines")
        tractogram = StatefulTractogram(streamlines, fod.nii, Space.RASMM)
        out_trk = self.outfile(f"{tract}/streamlines.trk")

        LOG.info(f" - Saving streamlines to {out_trk} ...")
        save_trk(tractogram, out_trk)

        fa = self.single_inimg("dtifit", "FA.nii.gz")
        FA_SEED_THRESH = 0.2 # FA threshold for seeding mask
        mask = self.single_inimg("brainmask", "b0_brain_mask.nii.gz")
        #SEED_DENSITY = 2 # seeds per voxel (integer)

        LOG.info(" - Preparing seeds from FA thresholded mask and stopping criterion...")
        os.makedirs(self.outfile(f"{tract}_2"), exist_ok=True)
        seed_mask = np.logical_and(fa.data > FA_SEED_THRESH, mask.data > 0)
        mask.save_derived(seed_mask.astype(np.uint8), self.outfile(f"{tract}_2/seed_mask.nii.gz"))
        #seeds = utils.seeds_from_mask(seed_mask, density=SEED_DENSITY, affine=dti.affine)
        sc = BinaryStoppingCriterion(seed_mask)

        streamline_generator = probabilistic_tracking(
            seeds,
            sc,
            dti.affine,
            sf=fod.data,
            random_seed=1,
            sphere=default_sphere,
            max_angle=20,
            step_size=0.2,
        )
        

        #LOG.info(" - Running LocalTracking")
        #streamlines_generator = LocalTracking(peaks, stopping_criterion, seeds=seeds, affine=dti.affine, step_size=STEP_SIZE)
        streamlines = list(streamline_generator)
        LOG.info(f" - Generated {len(streamlines)} streamlines")

        tractogram = StatefulTractogram(streamlines, dti.nii, Space.RASMM)
        out_trk = self.outfile(f"{tract}_2/streamlines.trk")
        LOG.info(f" - Saving streamlines to {out_trk} ...")
        save_trk(tractogram, out_trk)

class RegFSL(Module):
    """
    Registration using FSL FLIRT/FNIRT
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "reg_fsl", deps=["struc", "brainmask"], **kwargs)

    def process(self):
        fsldir = os.getenv("FSLDIR")
        if not fsldir:
            self.bad_data("FSLDIR environment variable not set")

        dti_b0 = self.single_inimg("brainmask", "mean_b0.nii.gz")
        if dti_b0 is None:
            self.no_data("No DTI mean B0 image found in brainmask/mean_b0.nii.gz")
        
        dti_b0_brain = self.single_inimg("brainmask", "b0_brain.nii.gz")
        if dti_b0_brain is None:
            self.no_data("No DTI brain image found in brainmask/b0_brain.nii.gz")

        t1 = self.single_inimg("struc/t1_orig", "t1.nii.gz")
        if t1 is None:
            self.no_data("No T1 image found in struc/t1_orig/t1.nii.gz")
        
        #t1_brain = self.single_inimg("struc/t1_orig", "t1_brain.nii.gz")
        #wmseg = self.single_inimg("struc/seg", "t1_biascorr_wm.nii.gz")

        LOG.info(" - Converting ANTs transforms to FSL format")
        self.runcmd([
            "c3d_affine_tool",
            "-ref", f"{fsldir}/data/standard/MNI152_T1_1mm.nii.gz",
            "-src", t1.fpath,
            "-itk", self.outfile("../struc/reg/t1_orig_to_std_lin.mat"),
            "-ras2fsl",
            "-o", self.outfile("t1_to_std_lin_fsl.mat"),
        ], logfile="c3d_t1_to_std_lin.log")
        self.runcmd([
            "c3d_affine_tool",
            "-ref", t1.fpath,
            "-src", dti_b0.fpath,
            "-itk", self.outfile("../dtireg/dti_to_t1_lin.mat"),
            "-ras2fsl",
            "-o", self.outfile("dti_to_t1_lin_fsl.mat"),
        ], logfile="c3d_dti_to_t1_lin_fsl.log")

        self.runcmd([
            "c3d",
            "-mcs", self.outfile("../struc/reg/t1_orig_to_std_nonlin.nii.gz"),
            "-oo", self.outfile("warp_x_tmp.nii.gz"), self.outfile("warp_y_tmp.nii.gz"), self.outfile("warp_z_tmp.nii.gz"),
        ], logfile="c3d_t1_to_std_nonlin.log")
        self.runcmd([
            "fslmaths",
            self.outfile("warp_y_tmp.nii.gz"),
            "-mul", "-1",
            self.outfile("warp_y_tmp.nii.gz"),
        ], logfile="fslmaths_t1_to_std_nonlin.log")
        self.runcmd([
            "fslmerge",
            "-t", self.outfile("t1_to_std_nonlin_fsl.nii.gz"),
            self.outfile("warp_x_tmp.nii.gz"), self.outfile("warp_y_tmp.nii.gz"), self.outfile("warp_z_tmp.nii.gz"),
        ], logfile="fslmerge_t1_to_std_nonlin.log")
        self.runcmd([
            "convertwarp",
            "--rel",
            "-r", f"{fsldir}/data/standard/MNI152_T1_1mm.nii.gz",
            "-m", self.outfile("t1_to_std_lin_fsl.mat"),
            "-w", self.outfile("t1_to_std_nonlin_fsl.nii.gz"),
            "-o", self.outfile("t1_to_std_nonlin_total_fsl.nii.gz"),
        ], logfile="convertwarp_t1_to_std_total.log")
        self.runcmd([
            "invwarp",
            "-r", t1.fpath,
            "-w", self.outfile("t1_to_std_nonlin_total_fsl.nii.gz"),
            "-o", self.outfile("std_to_t1_nonlin_total_fsl.nii.gz"),
        ], logfile="invwarp_std_to_t1_nonlin_total.log")

        self.runcmd([
            "convertwarp",
            "--relout",
            "-r", f"{fsldir}/data/standard/MNI152_T1_1mm.nii.gz",
            "-m", self.outfile("dti_to_t1_lin_fsl.mat"),
            "-w", self.outfile("t1_to_std_nonlin_total_fsl.nii.gz"),
            "-o", self.outfile("dti_to_std_nonlin_fsl.nii.gz"),
        ], logfile="convertwarp_dti_to_std.log")
        self.runcmd([
            "invwarp",
            "-w", self.outfile("dti_to_std_nonlin_fsl.nii.gz"),
            "-o", self.outfile("std_to_dti_nonlin_fsl.nii.gz"),
            "-r", dti_b0.fpath,
        ], logfile="invwarp_std_to_dti.log")

        #LOG.info(" - FLIRT DTI to T1 alignment")
        #self.runcmd([
        #    "flirt",
        #    "-in", dti_b0_brain.fpath,
        #    "-ref", t1_brain.fpath,
        #    "-dof", "6",
        #    "-omat", self.outfile("dti2t1_lin.mat"),
        #    "-out", self.outfile("dti2t1_lin.nii.gz"),
        #], logfile="flirt_init.log")

        #self.runcmd([
        #    "convert_xfm",
        #    "-omat", self.outfile("t12dti_lin.mat"),
        #    "-inverse", self.outfile("dti2t1_lin.mat"),
        #], logfile="invert_flirt_init.log")

        #LOG.info(" - FLIRT T1 to STD alignment")
        #self.runcmd([
        #    "flirt",
        #    "-interp", "spline",
        #    "-dof", "12",
        #    "-in", t1_brain.fpath,
        #    "-ref", f"{fsldir}/data/standard/MNI152_T1_1mm_brain.nii.gz",
        #    "-omat", self.outfile("t1_to_mni_linear.mat"),
        #    "-out", self.outfile("t1_to_mni_linear.nii.gz"),
        #], logfile="flirt_t1_to_std.log")
        
        #LOG.info(" - FNIRT T1 to STD non-linear alignment")
        #self.runcmd([
        #    "fnirt",
        #    "--interp", "spline",
        #    "--in", t1_brain.fpath,
        #    "--aff", self.outfile("t1_to_mni_linear.mat"),
        #    "--ref", f"{fsldir}/data/standard/MNI152_T1_1mm.nii.gz",
        #    "--cout", self.outfile("t1_to_mni_nonlin_coeff.nii.gz"),
        #    "--iout", self.outfile("t1_to_mni_nonlin.nii.gz"),
        #], logfile="fnirt_t1_to_std.log")

        #LOG.info(" - FLIRT BBR")
        #self.runcmd([
        #    "flirt",
        #    "-in", dti_b0.fpath,
        #    "-ref", t1.fpath,
        #    "-dof", "6",
        #    "-cost", "bbr",
        #    "-wmseg", wmseg.fpath,
        #    "-init", self.outfile("dti2t1_lin_init.mat"),
        #    "-omat", self.outfile("dti2t1_lin.mat"),
        #    "-out", self.outfile("dti2t1_lin.nii.gz"),
        #    "-schedule", f"{fsldir}/etc/flirtsch/bbr.sch",
        #], logfile="flirt_bbr.log")


class TractographyFSL(Module):
    """
    Anatomical tractography using FSL xtract / probtrackx2
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "tract_fsl", deps=["fibremod_fsl", "reg_fsl"], **kwargs)

    def process(self):
        with open(self.outfile("ptxopts.txt"), "w") as f:
            f.write("--savepaths --opathdir")

        self.runcmd([
            "xtract",
            "-bpx", self.outfile("../fibremod_fsl.bedpostX"),
            "-out", self.outfile(""),
            "-species", "HUMAN",
            "-stdwarp", os.path.abspath(self.outfile("../reg_fsl/std_to_dti_nonlin_fsl.nii.gz")), os.path.abspath(self.outfile("../reg_fsl/dti_to_std_nonlin_fsl.nii.gz")),
            "-ptx_options", self.outfile("ptxopts.txt"),
            "-gpu",
        ], logfile="xtract.log")
    #-p ${protdir} -str ${strlist} -queue imgpascalq

__version__ = "0.0.1"

NAME = "diffad"

MODULES = [
    # BrcPipeline(),
    BIDSDir(),
    #MRIQC(),
    StrucPreproc(),
    DtiPreproc(),
    DtiDenoise(),
    DtiUnring(),
    DtiDistCorr(),
    DtiBrainMask(),
    DtiReg(),
    DtiEddyCorrection(),
    DtiFitting(),
    FibreModelling(),
    Tractography(),
    RegFSL(),
    DtiFittingFSL(),
    FibreModellingFSL(),
    TractographyFSL(),
]


def add_options(parser):
    parser.add_argument("--b0-threshold", help="B0 threshold for diffusion data", type=float, default=50)
