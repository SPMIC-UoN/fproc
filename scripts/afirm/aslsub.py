import os, sys
import glob
import nibabel as nib
import numpy as np

BASEDIR="/spmstore/project/RenalMRI/afirm/output_20250130/"

for subjid in os.listdir(BASEDIR):
    subjdir = os.path.join(BASEDIR, subjid)
    for asl_subdir in ("fair_moco", "pcasl_moco"):
        asl_dir = os.path.join(subjdir, "fproc", asl_subdir)
        for fname in glob.glob(os.path.join(asl_dir, "*_moco.nii.gz")):
            print(f"Subtracting {fname}")
            nii = nib.load(fname)
            data = nii.get_fdata()
            if data.ndim != 4 or data.shape[-1] % 2 != 0:
                print(f"WARNING: Not LC data (shape {data.shape})")
                continue
            else:
                n = data.shape[-1] // 2
                c, l = data[..., :n], data[..., n:]
                diff = np.mean(c - l, axis=-1)
                nii2 = nib.Nifti1Image(diff, nii.header.get_best_affine(), nii.header)
                nii2.to_filename(fname.replace("_moco.nii", "_pwi.nii"))


