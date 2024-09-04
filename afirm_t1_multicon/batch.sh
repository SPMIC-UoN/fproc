#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH --job-name=afirm_t1_multicon
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imghmemq,imgcomputeq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
#NOTSBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-500
#SBATCH --output t1_multicon_logs/%A_%a.out
#SBATCH --error t1_multicon_logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
source activate renal_preproc

MODELDIR=/software/imaging/body_pipelines/trained_models/
OUTDIR=/spmstore/project/RenalMRI/afirm_t1_multicon/proc
#SLURM_ARRAY_TASK_ID=40

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

python pipelines/afirm_t1_multicon.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder seg --overwrite \
                                     --subjidx ${SLURM_ARRAY_TASK_ID} --t1-model=${MODELDIR}/kidney_t1_multicon_min_max.pt
