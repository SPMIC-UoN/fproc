#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=afirm_preproc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imghmemq,imgampereq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-501
#SBATCH --output scripts/afirm/logs/%A_%a.out
#SBATCH --error scripts/afirm/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load nnunetv2-img
module load conda-img
source activate renal_preproc

MODELDIR=/software/imaging/body_pipelines/trained_models/
OUTDIR=/spmstore/project/RenalMRI/afirm/output_20240807
#SLURM_ARRAY_TASK_ID=10

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"
sleep 10

fproc --pipeline pipelines/afirm_t2.py --input ${OUTDIR} --input-subfolder fsort_t2 --output ${OUTDIR} --output-subfolder fproc_t2 --overwrite --skip=t2\
                                 --subjidx ${SLURM_ARRAY_TASK_ID} \

