#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --job-name=afirm_preproc
##SBATCH --partition=imgcomputeq,imghmemq
##SBATCH --partition=imgampereq,imgpascalq,imgcomputeq,imghmemq,imgvoltaq
#SBATCH --partition=defq
##SBATCH --partition=imgampereq,imgvoltaq,imgpascalq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64g
##SBATCH --qos=img
##SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --array=0-5
#SBATCH --output scripts/afirm/logs/%A_%a.out
#SBATCH --error scripts/afirm/logs/%A_%a.err

module load renal-preproc-img
module load fsl-img/6.0.6.3
module load nnunetv2-img
module load conda-img
module load totalseg-img
module load traceseg-img
source activate renal_preproc

MODELDIR=/software/imaging/body_pipelines/trained_models/
BASEDIR=/spmstore/project/RenalMRI/afirm/
OUTDIR=${BASEDIR}/cb_test_fixes/output/
#SLURM_ARRAY_TASK_ID=0

echo "Processing subject ${SLURM_ARRAY_TASK_ID}"
sleep 5
export CUDA_VISIBLE_DEVICES=-1
fproc --pipeline pipelines/afirm.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder fproc --overwrite \
                                 --subjidx ${SLURM_ARRAY_TASK_ID} \
				 --cyst-masks ${BASEDIR}/fixed_cyst_masks \
				 --kidney-t1-model=${MODELDIR}/kidney_t1_molli_min_max.pt \
				 --kidney-t2w-model=${MODELDIR}/kidney_t2w_afirm_platnum_v02.h5 \
				 --seg-kidney-t2w-fix=${BASEDIR}/DEFINITIVETKVMASKS_RENAMED \
				 --seg-kidney-pelvis-fat-trace-fix=${BASEDIR}/FAT_FIXES \
				 --seg-kidney-cyst-trace=${BASEDIR}/TRACE_DATA_RENAMED \
				 --seg-organs-trace=${BASEDIR}/TRACE_ORGANS_RENAMED \
				 --t1-scale-factors=${BASEDIR}/afirm_t1_scale_factors_newids.csv \
				 --autoskip 

				 #--skip=pcasl_moco,fair_moco --force organ_dixon_radiomics
				 #--autoskip --force seg_kidney_whole,seg_kidney_pelvis_trace,seg_kidney_pelvis_trace_splitlr,wkv_ero_radiomics,wkv_ero_shape_metrics,wkv_radiomics,wkv_shape_metrics
				 # --force=temp_add_pelvis,seg_kidney_t2w_fix,seg_kidney_cyst_t2w_trace,seg_kidney_cyst_t2w_trace_splitlr,kidney_cyst_stats_trace_fixed,kidney_cyst_stats_trace_orig,cyst_volume_diffs,seg_organs_trace,seg_kidney_pelvis_trace,seg_kidney_pelvis_trace_splitlr,wkv_ero_radiomics,wkv_ero_shape_metrics,wkv_radiomics,wkv_shape_metrics \
				 #--skipdone=t1_molli,t1_molli_mdr,t1_molli_nomdr,t1_se_mdr,t1_se_nomdr,t1_se_mdr_step2,t2,t2star,mtr,b0,b1,dwi_moco,dwi_adc,pcasl_moco,fair_moco,seg_kidney_t1,seg_kidney_t1_mdr,seg_kidney_t1_nomdr,seg_kidney_t1_se,seg_kidney_cyst_t2w,seg_sat_dixon_cor,seg_liver_dixon_cor,seg_spleen_dixon_cor,seg_kidney_dixon_cor,seg_sat_dixon_ax,seg_liver_dixon_ax,seg_spleen_dixon_ax,seg_kidney_dixon_ax,seg_kidney_t2w,totalseg
