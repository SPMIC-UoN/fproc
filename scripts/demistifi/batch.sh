#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH --job-name=demistifi_fproc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imghmemq,imgcomputeq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80g
#SBATCH --qos=img
#NOTSBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-6500
#SBATCH --output scripts/demistifi/logs/%A_%a.out
#SBATCH --error scripts/demistifi/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
source activate renal_preproc

#OUTDIR=/imgshare/ukbiobank/demistifi_output_fullsets/NO_KIDNEY_DISEASE
#OUTDIR=/imgshare/ukbiobank/demistifi_output_fullsets/NO_KIDNEY_DATA
OUTDIR=/imgshare/ukbiobank/demistifi_output_fullsets/KIDNEY_DISEASE
#OUTDIR=/spmstore/project/RenalMRI/demistifi/output/Kidney_disease
#OUTDIR=/spmstore/project/RenalMRI/demistifi/output/Liver_disease
#OUTDIR=/spmstore/project/RenalMRI/demistifi/output/Pancreas_disease
#OUTDIR=/spmstore/project/RenalMRI/demistifi/output/Spleen_disease
SLURM_ARRAY_TASK_ID=3

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

fproc --pipeline pipelines/demistifi.py --input ${OUTDIR} --output ${OUTDIR} --output-subfolder fproc --overwrite \
                              --subjidx ${SLURM_ARRAY_TASK_ID} --skip=vat,asat

