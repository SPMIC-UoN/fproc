#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=mollihr_proc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imgpascalq,imgampereq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
##SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-5
#SBATCH --output scripts/mollihr/logs/%A_%a.out
#SBATCH --error scripts/mollihr/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
module load nnunetv2-img
module load ukbbseg-img
source activate renal_preproc

#SLURM_ARRAY_TASK_ID=0
echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

BASEDIR=/spmstore/project/RenalMRI
MODELDIR=/software/imaging/body_pipelines/trained_models/
INDIR=${BASEDIR}/mollihr/output_20241011
OUTDIR=${INDIR}

fproc --pipeline pipelines/mollihr.py --input ${INDIR} --input-subfolder fsort --output ${OUTDIR} --overwrite \
                              --subjidx ${SLURM_ARRAY_TASK_ID} --output-subfolder fproc \
                              --kidney-t1-model=${MODELDIR}/kidney_t1_molli_min_max.pt \
			      --skip=seg_liver_dixon,seg_spleen_dixon,seg_kidney_dixon,seg_kidney_t1

