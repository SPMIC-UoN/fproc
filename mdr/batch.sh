#!/bin/sh
#SBATCH --time=04:00:00
#SBATCH --job-name=mdr_compare
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
#SBATCH --output mdr/logs/%A_%a.out
#SBATCH --error mdr/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
source activate renal_preproc

MODELDIR=/software/imaging/body_pipelines/trained_models/
OUTDIR=/spmstore/project/RenalMRI/mdr/proc_20240522
#SLURM_ARRAY_TASK_ID=18

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

python pipelines/mdr_compare.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder compare --overwrite \
                                 --subjidx ${SLURM_ARRAY_TASK_ID} 

