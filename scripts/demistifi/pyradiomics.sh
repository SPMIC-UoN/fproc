#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH --job-name=demistifi_pyradiomics
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imghmemq,imgcomputeq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80g
#SBATCH --qos=img
#NOTSBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-100
#SBATCH --output demistifi/logs/%A_%a.out
#SBATCH --error demistifi/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
source activate renal_preproc

OUTDIR=/spmstore/project/RenalMRI/demistifi/pyradiomics_output
INDIR=/spmstore/project/RenalMRI/demistifi/pyradiomics_data/Mrker_T1
#SLURM_ARRAY_TASK_ID=1

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

python pipelines/demistifi_pyradiomics.py --input ${INDIR} --output ${OUTDIR} --overwrite \
                              --subjidx ${SLURM_ARRAY_TASK_ID} 

