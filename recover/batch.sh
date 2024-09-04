#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=recover_preproc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imghmemq,imgcomputeq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
#NOTSBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-25
#SBATCH --output recover/logs/%A_%a.out
#SBATCH --error recover/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
source activate renal_preproc

MODELDIR=/spmstore/project/RenalMRI/trained_models/
OUTDIR=/spmstore/project/RenalMRI/recover/proc
SLURM_ARRAY_TASK_ID=16

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

python pipelines/renal_preproc.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder preproc --overwrite \
                                 --subjidx ${SLURM_ARRAY_TASK_ID} \
                                 --t2star-method=all --skip=t1w,mtr,t1,t2star \
                                 --t2w-model=${MODELDIR}/t2w_seg.h5 --t2w-fixed-masks=/spmstore/project/RenalMRI/recover/fixed_masks_touse 


                                 #--skip=t1,t2star,t2w \
                                 #--skip=t2star,t1,t2w,t1w,b0,b1,mtr,align,resample,t1_clean \
