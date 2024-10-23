#!/bin/sh
#SBATCH --time=08:00:00
#SBATCH --job-name=tk21_proc
#SBATCH --partition=imgcomputeq,imghmemq
#NOTSBATCH --partition=imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
#NOTSBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-50
#SBATCH --output scripts/tk21/logs/%A_%a.out
#SBATCH --error scripts/tk21/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
#module load nnunetv2-img
source activate renal_preproc

#SLURM_ARRAY_TASK_ID=9
echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

MODELDIR=/software/imaging/body_pipelines/trained_models/
BASEDIR=/spmstore/project/RenalMRI/tk21
OUTDIR=${BASEDIR}/proc_20240904

fproc --pipeline pipelines/tk21.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder fproc_full --overwrite \
                           --subjidx ${SLURM_ARRAY_TASK_ID} \
			   --kidney-t1-model=${MODELDIR}/kidney_t1_molli_min_max.pt \
			   --kidney-t2w-model=${MODELDIR}/t2w_seg.h5 \
                           --kidney-masks ${BASEDIR}/GOSH_masks \
			   --skip=t1_kidney,seg_kidney_t1,seg_kidney_t2w,t2star \
                           --t2star-method=all \


