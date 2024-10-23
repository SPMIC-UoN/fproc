#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=mlrepeat_proc
#NOTSBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imgpascalq,imgampereq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --qos=img
##SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-70
#SBATCH --output scripts/mlrepeat/logs/%A_%a.out
#SBATCH --error scripts/mlrepeat/logs/%A_%a.err

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
INDIR=${BASEDIR}/mlrepeat/proc_20240910
OUTDIR=${INDIR}

fproc --pipeline pipelines/mlrepeat.py --input ${INDIR} --input-subfolder fsort --output ${OUTDIR} --overwrite \
                              --subjidx ${SLURM_ARRAY_TASK_ID} --output-subfolder fproc \
                              --kidney-t1-model=${MODELDIR}/kidney_t1_molli_min_max.pt \
			      --kidney-t2w-model=${MODELDIR}/t2w_seg.h5 \
			      --pancreas-masks=${BASEDIR}/mlrepeat/pancreas_masks \
			      --liver-masks=${BASEDIR}/mlrepeat/liver_masks \
			      --skip=seg_kidney_t1,seg_pancreas_ethrive,seg_liver_dixon,seg_spleen_dixon,seg_kidney_t2w,seg_kidney_dixon


