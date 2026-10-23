#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --job-name=afirm_preproc
##SBATCH --partition=imgcomputeq,imghmemq
##SBATCH --partition=imgampereq,imgpascalq,imgcomputeq,imghmemq,imgvoltaq
#SBATCH --partition=defq
##SBATCH --partition=imgampereq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32g
##SBATCH --qos=img
##SBATCH --gres=gpu:1
#SBATCH --export=NONE
##SBATCH --array=0-600
#SBATCH --output scripts/afirm/logs/%A_%a.out
#SBATCH --error scripts/afirm/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load nnunetv2-img
module load conda-img
source activate renal_preproc

MODELDIR=/software/imaging/body_pipelines/trained_models/
BASEDIR=/spmstore/project/RenalMRI/afirm/
OUTDIR=${BASEDIR}/oneoff
SLURM_ARRAY_TASK_ID=0

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"
sleep 5
export CUDA_VISIBLE_DEVICES=-1
echo $CUDA_VISIBLE_DEVICES
fproc --pipeline pipelines/afirm.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder fproc --overwrite \
                                 --subjidx ${SLURM_ARRAY_TASK_ID} \
				 --cyst-masks ${BASEDIR}/fixed_cyst_masks \
				 --kidney-t1-model=${MODELDIR}/kidney_t1_molli_min_max.pt \
				 --kidney-t2w-model=${MODELDIR}/t2w_seg.h5 \
				 --seg-kidney-t2w-fix=${BASEDIR}/fixed_masks_new \
				 --t1-scale-factors=${BASEDIR}/afirm_t1_scale_factors.csv \


