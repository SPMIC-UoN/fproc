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
#SBATCH --array=0-100
#SBATCH --output tk21/logs/%A_%a.out
#SBATCH --error tk21/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
module load nnunetv2-img
source activate renal_preproc

#SLURM_ARRAY_TASK_ID=9
echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

MODELDIR=/software/imaging/body_pipelines/trained_models/
INDIR=/spmstore/project/RenalMRI/tk21/proc
OUTDIR=/spmstore/project/RenalMRI/tk21/proc

python pipelines/tk21_t2.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder fproc --overwrite \
                           --subjidx ${SLURM_ARRAY_TASK_ID} 

