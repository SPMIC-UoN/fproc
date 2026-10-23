#!/bin/sh
#SBATCH --time=06:00:00
#SBATCH --job-name=afirm_preproc
##SBATCH --partition=imgcomputeq,imghmemq
#SBATCH --partition=imgampereq,imgpascalq,imgcomputeq,imghmemq
##SBATCH --partition=imgampereq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32g
#SBATCH --qos=img
##SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-600
#SBATCH --output scripts/afirm/logs/%A_%a.out
#SBATCH --error scripts/afirm/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load nnunetv2-img
module load conda-img
source activate renal_preproc

MODELDIR=/software/imaging/body_pipelines/trained_models/
BASEDIR=/spmstore/project/RenalMRI/afirm/
OUTDIR=${BASEDIR}/output_20250116
#SLURM_ARRAY_TASK_ID=13

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"
sleep 5
echo $CUDA_VISIBLE_DEVICES
fproc --pipeline pipelines/afirm_t2.py --input ${OUTDIR} --input-subfolder fsort_t2 --output ${OUTDIR} --output-subfolder fproc_t2 --overwrite \
                                 --subjidx ${SLURM_ARRAY_TASK_ID} \
				 --cyst-masks ${BASEDIR}/fixed_cyst_masks \
				 --kidney-t1-model=${MODELDIR}/kidney_t1_molli_min_max.pt \
				 --t1-scale-factors=${BASEDIR}/afirm_t1_scale_factors.csv \
				 --skip=seg_kidney_t1,t2,seg_kidney_cyst_t2w,seg_sat_dixon_cor,seg_spleen_dixon_cor,seg_liver_dixon_cor \

