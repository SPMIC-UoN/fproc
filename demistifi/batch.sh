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
#SBATCH --array=0-1300
#SBATCH --output demistifi/logs/%A_%a.out
#SBATCH --error demistifi/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
source activate renal_preproc

OUTDIR=/spmstore/project/RenalMRI/demistifi/output_newsubjs/NO_KIDNEY_DATA
#OUTDIR=/spmstore/project/RenalMRI/demistifi/output/Kidney_disease
#OUTDIR=/spmstore/project/RenalMRI/demistifi/output/Liver_disease
#OUTDIR=/spmstore/project/RenalMRI/demistifi/output/Pancreas_disease
#OUTDIR=/spmstore/project/RenalMRI/demistifi/output/Spleen_disease
#SLURM_ARRAY_TASK_ID=1300

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

python pipelines/demistifi.py --input ${OUTDIR} --output ${OUTDIR} --output-subfolder fproc --overwrite \
                              --subjidx ${SLURM_ARRAY_TASK_ID} 

