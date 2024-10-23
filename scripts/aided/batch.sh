#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=aided_fproc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imgvoltaq,imgpascalq,imgampereq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-30
#SBATCH --output aided/logs/%A_%a.out
#SBATCH --error aided/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
module load nnunetv2-img
source activate renal_preproc

SLURM_ARRAY_TASK_ID=25
echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

MODELDIR=/software/imaging/body_pipelines/trained_models/
INDIR=/gpfs01/spmstore/project/RenalMRI/aided/proc
OUTDIR=/gpfs01/spmstore/project/RenalMRI/aided/proc

python pipelines/resus_proc.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --overwrite \
                              --subjidx ${SLURM_ARRAY_TASK_ID} --output-subfolder fproc --debug 

