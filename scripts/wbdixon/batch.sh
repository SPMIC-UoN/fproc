#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=wbdixon_proc
#SBATCH --partition=imgpascalq,imgampereq,imgcomputeq,imghmemq,imgvoltaq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
##SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-100
#SBATCH --output scripts/wbdixon/logs/%A_%a.out
#SBATCH --error scripts/wbdixon/logs/%A_%a.err

module load conda-img
module load leg-seg-dixon-img
module load nnunetv2-img
source activate renal_preproc

#SLURM_ARRAY_TASK_ID=10
echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

BASEDIR=/spmstore/project/RenalMRI
MODELDIR=/software/imaging/body_pipelines/trained_models/
INDIR=${BASEDIR}/wbdixon/data
OUTDIR=${BASEDIR}/wbdixon/output_20241014

fproc --pipeline pipelines/wbdixon.py --input ${INDIR} --output ${OUTDIR} --overwrite \
                              --subjidx ${SLURM_ARRAY_TASK_ID} --output-subfolder fproc \
                              --leg-dixon-model=${MODELDIR}/leg_dixon \
#			      --skip=seg_leg_dixon


