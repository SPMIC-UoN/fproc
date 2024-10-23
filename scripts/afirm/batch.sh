#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=afirm_preproc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imghmemq,imgcomputeq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
#NOTSBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-550
#SBATCH --output afirm_logs/%A_%a.out
#SBATCH --error afirm_logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
source activate renal_preproc

MODELDIR=/software/imaging/body_pipelines/trained_models/
OUTDIR=/spmstore/project/RenalMRI/afirm/proc_20240730
SLURM_ARRAY_TASK_ID=200

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

python pipelines/renal_preproc.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder fproc --overwrite \
                                 --subjidx ${SLURM_ARRAY_TASK_ID} \
                                 --t2star-method=all \
                                 --skip=t2star,t1w,b0,b1,mtr \
                                 --t2w-model=${MODELDIR}/t2w_seg.h5 --t2w-fixed-masks=/spmstore/project/RenalMRI/afirm/fixed_masks 


                                 #--skip=t1,t2star,t2w \
                                 #--skip=t2star,t1,t2w,t1w,b0,b1,mtr,align,resample,t1_clean \
