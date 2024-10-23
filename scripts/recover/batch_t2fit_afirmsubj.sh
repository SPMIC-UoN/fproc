#!/bin/sh
#SBATCH --time=08:00:00
#SBATCH --job-name=tk21_proc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-50
#SBATCH --output recover/logs/%A_%a.out
#SBATCH --error recover/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
module load nnunetv2-img
source activate renal_preproc

SLURM_ARRAY_TASK_ID=0
echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

MODELDIR=/software/imaging/body_pipelines/trained_models/
INDIR=/spmstore/project/RenalMRI/recover/afirmsubj
OUTDIR=/spmstore/project/RenalMRI/recover/afirmsubj/

python pipelines/tk21_t2.py --input ${OUTDIR} --output ${OUTDIR} --input-subfolder fsort --overwrite \
                            --subjidx ${SLURM_ARRAY_TASK_ID} 
