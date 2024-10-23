#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=bariatric_proc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imgpascalq,imgampereq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-70
#SBATCH --output bariatric/logs/%A_%a.out
#SBATCH --error bariatric/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load conda-img
module load nnunetv2-img
source activate renal_preproc

#SLURM_ARRAY_TASK_ID=30
echo "Processing subject ${SLURM_ARRAY_TASK_ID}"

BASEDIR=/spmstore/project/RenalMRI/bariatric
MODELDIR=/software/imaging/body_pipelines/trained_models/
INDIR=${BASEDIR}/proc_20240813
OUTDIR=${INDIR}

fproc --pipeline pipelines/bariatric.py --input ${INDIR} --input-subfolder fsort --output ${OUTDIR} --overwrite \
                              --subjidx ${SLURM_ARRAY_TASK_ID} --output-subfolder fproc \
                              --kidney-t1-model=${MODELDIR}/kidney_t1_molli_min_max.pt \
			      --pancreas-masks=${BASEDIR}/pancreas_masks \
			      --liver-masks=${BASEDIR}/liver_masks \
			      --sat-masks=${BASEDIR}/sat_masks \
			      --se-t1-maps=${BASEDIR}/se_t1_maps_renamed \
			      --adc-maps=${BASEDIR}/adc_maps_renamed \

