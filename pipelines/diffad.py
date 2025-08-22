import glob
import logging
import os
import shutil

import numpy as np
import scipy.ndimage
import ants

from fsort.image_file import ImageFile
from fproc.module import Module
from fproc.modules import segmentations, seg_postprocess, statistics, maps, regrid

LOG = logging.getLogger(__name__)

WM = [2, 7, 28, 30, 31, 78, 100, 41, 60, 62, 63, 79, 108, 109, 117, 77, 85, 192, 251, 252, 253, 254, 255, 46, 16]
GM = [3, 8, 10, 11, 12, 13, 17, 18, 26, 96, 101, 102, 103, 104, 105, 106, 107, 42, 49, 50, 51, 52, 53, 54, 58, 97, 110, 111, 112, 113, 114, 115, 116, 80, 47]
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

# ##################
# ######### BRC diffusion processing
# ##################
# # Ideally we would have the whole BRC diffusion pipeline outside of FSL (in dipy???)
# # topup: to be confirmed - https://pmc.ncbi.nlm.nih.gov/articles/PMC6906182/pdf/fninf-13-00076.pdf looks like a good starting point. It suggests DR-BUDDI in TORTOISE (https://github.com/QMICodeBase/TORTOISEV4, https://tortoise.nibib.nih.gov/tortoise) - need to explore license
# # eddy: Niifreeze https://github.com/nipreps/nifreeze
# # registration: ANTs? - I'm less familiar with EPI registration tools outside of FSL
# # DTIFIT and CUDIMOT NODDI: there are many options in dipy for basic diffusion modeling (if we can get tensor modelling working in the pipeline, that would be a good first step. We can worry about NODDI later if it's less obvious)

# # denoising: we don't perform it in our current data as UKB data doesn't do it as standard but we would want to include this in the future (NORDIC or MPPCA) - again, several options available through dipy

# # dMRI QC: could use dipy or mriQC for basic QC metrics

# # overall, several steps can be completed using e.g. dipy: https://dipy.org/. We need to double check what dipy is doing under the hood to make sure it's not calling FSL! FSL isn't mentioned in their dependencies so I assume it doesn't
# # MRtrix is another option, but much of their diffusion preprocessing steps are calling FSL

# cd ${procdir}/${subid}/proc
# ees=`grep "EchoSpacing" ${procdir}/${subid}/raw/'Axial_MB_DTI_PA_(MSV21)_Si'.json`
# ees=${ees#*: }
# ees=${ees%%,*}
# if [[ -z "$ees" ]]; then ees=0.00055; fi
# rm -rf ${procdir}/${subid}/proc/${subid}/analysis/dMRI
# dMRI_preproc.sh --input ${procdir}/${subid}/raw/'Axial_MB_DTI_PA_(MSV21)_Si'.nii.gz --input_2 ${procdir}/${subid}/raw/'Axial_MB_DTI_AP_(MSV21)_Si'.nii.gz --path ${procdir}/${subid}/proc --subject ${subid} --pe_dir 2 --echospacing ${ees} --p_im 1 --slspec ${procdir}/${subid}/raw/'Axial_MB_DTI_PA_(MSV21)_Si'.json --dtimaxshell 1000 --qc --reg --noddi

# ##################
# ######### advanced diffusion processing (not included in the BRC pipeline)
# ##################
# module load fsl-img/6.0.7.x

# cd ${procdir}/${subid}/proc

# # crossing-fibre modelling with bedpostX
# # could be replaced by CSD e.g. https://docs.dipy.org/stable/examples_built/reconstruction/reconst_csd.html
# fsl_sub -N bpx_${subid} --coprocessor=cuda -q imgvoltaq -R 20 -T 120 bedpostx_gpu ${procdir}/${subid}/proc/${subid}/analysis/dMRI/processed/data --nf=3 --fudge=1 --bi=3000 --nj=1250 --se=25 --model=2 --cnonlinear


# # WM tract segmentation with xtract
# # xtract in essence just needs NIFTIs (protocols), warps field and a warp tool (equivalent to applywarp), and a tractography tool (equivalent to probtrackx)
# # warps/registrations - probtrackx handles standard space to diffusion space transforms internally. Need to assess how e.g. dipy would deal with standard space protocols
# # dipy has a few tractography options, including probabilistic tractography: https://docs.dipy.org/stable/examples_built/fiber_tracking/tracking_probabilistic.html#sphx-glr-examples-built-fiber-tracking-tracking-probabilistic-py
# # I think we should be able to take the xtract protocols (as released on our local github) and use them with a dipy tractography tools to effectively perform the job of xtract. I would just then need to re-write xtract_stats to get tract-wise summary measures (a simple task)
# xtract -bpx ${procdir}/${subid}/proc/${subid}/analysis/dMRI/processed/data.bedpostX -out ${procdir}/${subid}/proc/${subid}/analysis/dMRI/processed/xtract_6.0.7.15 -stdwarp ${procdir}/${subid}/proc/${subid}/analysis/dMRI/preproc/reg/std_2_diff_warp_coeff.nii.gz ${procdir}/${subid}/proc/${subid}/analysis/dMRI/preproc/reg/diff_2_std_warp_coeff.nii.gz -species HUMAN -p ${protdir} -str ${strlist} -gpu -queue imgpascalq

# # xtract_stats to get tract-wise summary features (tract volume, FA, MD, ISOVF, etc...)
# fsl_sub -N xsa_${subid} -R 10 -T 30 xtract_stats -xtract ${procdir}/${subid}/proc/${subid}/analysis/diffusion/xtract_6.0.7.15 -warp ${procdir}/${subid}/proc/${subid}analysis/dMRI/processed/data/data.dti/dti_FA.nii.gz ${procdir}/${subid}/proc/${subid}/analysis/diffusion/MNI_to_dti_FA_warp.nii.gz -d ${procdir}/${subid}/proc/${subid}/analysis/dMRI/processed/data/data.dti/dti_ -meas vol,prob,length,FA,MD,MO -out ${procdir}/${subid}/proc/${subid}/analysis/diffusion/xtract_6.0.7.15/xtract_stats_dti.csv

# fsl_sub -N xsb_${subid} -R 10 -T 30 xtract_stats -xtract ${procdir}/${subid}/proc/${subid}/analysis/diffusion/xtract_6.0.7.15 -warp ${procdir}/${subid}/proc/${subid}/analysis/dMRI/processed/data/data.dti/dti_FA.nii.gz ${procdir}/${subid}/proc/${subid}/analysis/diffusion/std_2_diff_warp_coeff.nii.gz -d ${procdir}/${subid}/proc/${subid}/analysis/dMRI/processed/data/data.noddi/NODDI_ -meas ICVF,ISOVF,OD -out ${procdir}/${subid}/proc/${subid}/analysis/diffusion/xtract_6.0.7.15/xtract_stats_NODDI.csv

class BrcPipeline(Module):
    """
    BRC Pipeline
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
        self.runcmd([
                "struc_preproc.sh",
                "--input", t1.fpath,
                "--t2", t2.fpath,
                "--path", self.outdir,
                "--subject", "",
                "--qc"
            ],
            logfile="brc_struc.log",
        )

class StrucPreproc(Module):
    """
    BRC Pipeline-like Structural Preprocessing but without FSL
    """

    def __init__(self, **kwargs):
        Module.__init__(self, "struc", **kwargs)
        self.spaces = {"t2_orig": {}, "t1_orig" : {}, "t1" : {}, "std" : {}}
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
        self.runcmd([
                "mri_synthstrip",
                "-i", self._fname(name, space),
                "-o", self.outfile(f"{space}/{name}_brain.nii.gz"),
                "-m", self.outfile(f"{space}/{name}_brain_mask.nii.gz"),
            ],
            logfile="mri_synthstrip_t1.log"
        )
        self._fromfile(f"{name}_brain", space)
        self._fromfile(f"{name}_brain_mask", space)

    def _get_stdref(self):
        fsl_stdref_path = f"{os.environ.get('FSLDIR', '')}/data/standard/MNI152_T1_1mm.nii.gz"
        stdref_path = self.kwargs.get("stdref", fsl_stdref_path)
        LOG.info(" - Using standard space reference: %s", stdref_path)
        stdref = ImageFile(stdref_path, warn_json=False)
        self._save(stdref, "ref", "std")
        fsl_stdref_brain_mask_path = f"{os.environ.get('FSLDIR', '')}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
        stdref_brain_mask_path = self.kwargs.get("stdref_brain", fsl_stdref_brain_mask_path)
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
            deface_mask_ants,
            self.spaces["std"]["ref_ants"],
            interp_type="genericLabel"
        )
        self._save_ants(deface_mask_std_ants, "deface_mask", "std")


    def _reg(self, name, from_space, to_space, type_of_transform='Affine', initial_transform=None):
        img_ants = self.spaces[from_space][f"{name}_ants"]
        ref_ants = self.spaces[to_space]["ref_ants"]
        reg_result = ants.registration(
            ref_ants,
            img_ants,
            initial_transform=initial_transform,
            type_of_transform=type_of_transform
        )

        self._save_ants(reg_result["warpedmovout"], name, to_space)
        if from_space not in self.reg:
            self.reg[from_space] = {}
        if to_space not in self.reg:
            self.reg[to_space] = {}
        #self.reg[from_space][to_space] = reg_result["fwdtransforms"]
        #self.reg[to_space][from_space] = reg_result["invtransforms"]
        
        if type_of_transform == 'SyN':
            shutil.copyfile(reg_result["fwdtransforms"][0], self.outfile(f"reg/{from_space}_to_{to_space}_nonlin.nii.gz"))
            shutil.copyfile(reg_result["fwdtransforms"][1], self.outfile(f"reg/{from_space}_to_{to_space}_lin.mat"))
            shutil.copyfile(reg_result["invtransforms"][1], self.outfile(f"reg/{to_space}_to_{from_space}_nonlin.nii.gz"))
            shutil.copyfile(reg_result["invtransforms"][0], self.outfile(f"reg/{to_space}_to_{from_space}_lin.mat"))
            self.reg[from_space][to_space] = [
                self.outfile(f"reg/{from_space}_to_{to_space}_nonlin.nii.gz"),
                self.outfile(f"reg/{from_space}_to_{to_space}_lin.mat")
            ]
            self.reg[to_space][from_space] = [
                self.outfile(f"reg/{to_space}_to_{from_space}_lin.mat"),
                self.outfile(f"reg/{to_space}_to_{from_space}_nonlin.nii.gz")
            ]
        else:
            shutil.copyfile(reg_result["fwdtransforms"][0], self.outfile(f"reg/{from_space}_to_{to_space}_lin.mat"))
            shutil.copyfile(reg_result["invtransforms"][0], self.outfile(f"reg/{to_space}_to_{from_space}_lin.mat"))
            self.reg[from_space][to_space] = [self.outfile(f"reg/{from_space}_to_{to_space}_lin.mat")]
            self.reg[to_space][from_space] = [self.outfile(f"reg/{to_space}_to_{from_space}_lin.mat")]

    def _transform(self, name, from_space, to_space, is_roi=False):
        src = self.spaces[from_space][f"{name}_ants"]
        dest_ref = self.spaces[to_space]["ref_ants"]
        transform = self.reg[from_space][to_space]
        if transform == "identity":
            dest = ants.resample_image_to_target(
                src,
                dest_ref,
                interp_type="genericLabel" if is_roi else "linear"
            )
        else:
            dest = ants.apply_transforms(
                dest_ref,
                src,
                transform,
                interpolator="genericLabel" if is_roi else "linear"
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
        cropped_affine[:3, 3] = img.affine[:3, 3] + np.dot(img.affine[:3, :3], [bbox[i].start for i in range(3)])
        import nibabel as nib
        cropped_nii = nib.Nifti1Image(cropped_data, affine=cropped_affine, header=img.header)
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
        head_img.save_derived(defaced_data, self.outfile(f"{space}/{name}_defaced.nii.gz"))
        self._fromfile(f"{name}_defaced", space)

    def _biasfield(self, name, mask_name, space):
        biasfield = ants.n4_bias_field_correction(
            self.spaces[space][f"{name}_ants"],
            self.spaces[space][f"{mask_name}_ants"],
            return_bias_field=True,
            shrink_factor=4
        )
        biasfield = biasfield.numpy()
        biasfield = biasfield / np.mean(biasfield)
        biasfield[np.isclose(biasfield, 0, atol=1e-3)] = 1  # Avoid division by zero
        biasfield[~np.isfinite(biasfield)] = 1  # Remove any non-finite values
        self.spaces[space][name].save_derived(biasfield, self.outfile(f"{space}/biasfield.nii.gz"))
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
        self.runcmd([
               "mri_synthseg",
               "--i", self.outfile(f"{space}/{name}.nii.gz"),
               "--o", self.outfile(f"seg/{name}_seg.nii.gz"),
               "--vol", self.outfile(f"seg/{name}_seg.csv"),
               "--cpu",
               "--robust",
           ],
           logfile="mri_synthseg.log"
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
        self._reg("t1", "t1_orig", "std", type_of_transform='Affine')
        
        LOG.info(" - Performing nonlinear registration of T1 to standard space reference")
        self._reg("t1", "t1_orig", "std", type_of_transform='SyN', initial_transform=self.reg["t1_orig"]["std"])

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
        self._reg("t2", "t2_orig", "t1_orig", type_of_transform='Rigid')
        self.reg["t2_orig"]["std"] = self.reg["t2_orig"]["t1_orig"] + self.reg["t1_orig"]["std"]
        self.reg["std"]["t2_orig"] = self.reg["std"]["t1_orig"] + self.reg["t1_orig"]["t2_orig"]
        
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

class DistCorr(Module):

    def __init__(self, **kwargs):
        Module.__init__(self, "distcorr", **kwargs)

    def process(self):
        dti_pa = self.single_inimg("dti_pa", "dti_pa.nii.gz")
        dti_ap = self.single_inimg("dti_ap", "dti_ap.nii.gz")
        if dti_pa is None:
            self.no_data("No DTI PA data found in dti_pa")
        if dti_ap is None:
            self.no_data("No DTI AP data found in dti_ap")

        t2 = self.single_inimg("flair", "flair.nii.gz")
        if t2 is None:
            self.no_data("No T2 data found in flair")

        LOG.info("Running TORTOISE for distortion correction")
        retval = self.runcmd([
                "TORTOISEProcess",
                "--up_data", dti_pa.fpath,
                "--up_json", dti_pa.json_fpath,
                "--down_data", dti_ap.fpath,
                "--denoising", "off",
                "--gibbs", "",
                "-c", "off",
                "--s2v", "0",
                "--repol", "0",
                "-s", t2.fpath,
                "-o", self.outfile("tortoise.nii.gz"),
                "-t", self.outfile("temp")
            ],
            logfile="tortoise.log"
        )
        if retval != 0:
            self.bad_data(f"TORTOISEProcess failed with return code {retval}")

class DiffPreproc(Module):
    """
    BRC Pipeline-like Diffusion Preprocessing but without FSL
    """
    def __init__(self, **kwargs):
        Module.__init__(self, "diff", **kwargs)

    def process(self):
        src = self.kwargs.get("src", "dti")
        glob = self.kwargs.get("glob", "dti*.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src} with glob {glob}")
        
        es = dti.effectiveechospacing
        if es is None:
            LOG.warning("No effective echo spacing found in the diffusion data. Using default value of 0.00055 seconds.")
            es = 0.00055
        
        encoding_dir = dti.phaseencodingdirection
        if encoding_dir is None:
            LOG.warning("No phase encoding direction found in the diffusion data. Using default value of 2 (PA).")
            encoding_dir = 2
        elif encoding_dir == "j-":
            encoding_dir = -2 # AP
        elif encoding_dir == "j":
            encoding_dir = 2 # PA
        elif encoding_dir == "i":
            encoding_dir = 1 # LR
        elif encoding_dir == "i-":
            encoding_dir = -1 # RL
        else:
            LOG.warning(f"Unknown phase encoding direction '{encoding_dir}' found in the diffusion data. Using default value of 2 (PA).")
            encoding_dir = 2

        # Compute echo spacing and total readout time
        pe_steps = dti.phaseencodingsteps
        if pe_steps is None:
            LOG.warning("No phase encoding steps found in the diffusion data. Using NDIM-1 from phase encoding direction.")
            if encoding_dir in [1, -1]:  # LR or RL
                pe_steps = dti.shape[0]
            elif encoding_dir in [2, -2]:  # AP or PA
                pe_steps = dti.shape[1]
        else:
            LOG.info(f"Using phase encoding steps from the data: {pe_steps}")
        
        grappa = 1
        total_readout_time = es * (pe_steps - 1) / grappa
        LOG.info(f"Total readout time calculated as: {total_readout_time:.6f} seconds")

        # Denoising
        # Unringing
        # Intensity normalisation

        # conversion of all input files to the same datatype to ensure consistency

        # Mean at each timepoint
        # ${FSLDIR}/bin/fslmaths ${entry} -Xmean -Ymean -Zmean ${basename}_mean
        bvals = dti.bvals
        LOG.info(f"Using b-values: {bvals}")
        mean_per_bval = np.mean(dti.data, axis=(0, 1, 2))

        
        #Posbvals=`cat ${basename}.bval`
        #mcnt=0

        # Mean intensity at each b0
        #for i in ${Posbvals} #extract all b0s for the series
        #do
        #    cnt=`$FSLDIR/bin/zeropad $mcnt 4`
        #    if [ $i -lt ${b0maxbval} ]; then
        #        $FSLDIR/bin/fslroi ${basename}_mean ${basename}_b0_${cnt} ${mcnt} 1
        #    fi
        #    mcnt=$((${mcnt} + 1))
        #done

        #${FSLDIR}/bin/fslmerge -t ${basename}_mean `echo ${basename}_b0_????.nii*`
        #${FSLDIR}/bin/fslmaths ${basename}_mean -Tmean ${basename}_mean #This is the mean baseline b0 intensity for the series
        #${FSLDIR}/bin/imrm ${basename}_b0_????

        #if [ ${entry_cnt} -eq 0 ]; then      #Do not rescale the first series
        #    rescale=`${FSLDIR}/bin/fslmeants -i ${basename}_mean`
        #else
        #    scaleS=`${FSLDIR}/bin/fslmeants -i ${basename}_mean`
        #    ${FSLDIR}/bin/fslmaths ${basename} -mul ${rescale} -div ${scaleS} ${basename}_new
        #    ${FSLDIR}/bin/imrm ${basename}   #For the rest, replace the original dataseries with the rescaled one
        #    ${FSLDIR}/bin/immv ${basename}_new ${basename}
        #fi

        #entry_cnt=$((${entry_cnt} + 1))
    #${FSLDIR}/bin/imrm ${basename}_mean
        
        
class DtiFitting(Module):
    """
    Fitting of diffusion tensor imaging (DTI) data
    """
    def __init__(self, **kwargs):
        Module.__init__(self, "dtifit", **kwargs)

    def process(self):
        # FIXME no preprocessing yet
        from dipy.core.gradients import gradient_table
        from dipy.reconst.dti import TensorModel, lower_triangular, fractional_anisotropy

        src = self.kwargs.get("src", "dti_pa")
        glob = self.kwargs.get("glob", "dti*.nii.gz")
        dti = self.single_inimg(src, glob)
        if dti is None:
            self.no_data(f"No diffusion data found in {src} with glob {glob}")
        
        print(dti.fpath)
        print(dti.bval)
        print(dti.bvec)
        grad_table = gradient_table(
            np.array(dti.bval),
            np.array(dti.bvec),
            #b0_threshold=50  # Threshold for b0 values
        )
        LOG.info(f"Using gradient table: {grad_table}")

        # FIXME mask data
        LOG.info("Fitting diffusion tensor model to the data")
        tensor_model = TensorModel(grad_table)
        tensor_fit = tensor_model.fit(dti.data)
        tensor_vals = lower_triangular(tensor_fit.quadratic_form)

        LOG.info("Computing anisotropy measures (FA, MD, RGB)")
        FA = fractional_anisotropy(tensor_fit.evals)
        dti.save_derived(FA, self.outfile("FA.nii.gz"))
        dti.save_derived(tensor_fit.md, self.outfile("MD.nii.gz"))
        for idx in range(3):
            print(tensor_fit.evals[..., idx].shape)
            dti.save_derived(tensor_fit.evals[..., idx], self.outfile(f"L{idx+1}.nii.gz"))
            dti.save_derived(tensor_fit.evecs[..., idx], self.outfile(f"V{idx+1}.nii.gz"))
        #dti.save_derived(tensor_fit.model_S0, self.outfile("S0.nii.gz"))

__version__ = "0.0.1"

NAME = "diffad"

MODULES = [
    #BrcPipeline(),
    StrucPreproc(),
    DistCorr(),
    DtiFitting(),
]

def add_options(parser):
    pass
