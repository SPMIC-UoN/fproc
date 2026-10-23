#!/bin/sh
#SBATCH --time=72:00:00
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
##SBATCH --array=0-750

# enhanced dicom
##SBATCH --array=45,46,47,48,49,50,52,55,58,60,62,64,68,69,71,73,75,77,81,83,85,87,89,91,93,95,98,100,103,105,106,107,109,111,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,165,167,168,169,171,172,173,174,175,176,177,178,179,180,181,182,184,185,186,187,188,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,211,213,214,215,216,217,218,219,220,221,222,223,224,226,471,472,473,474,475,476,477,478,479,480,481,482,483,485,486,487,488,489,490,491,492,493,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,519,520,521,522,523,524,526,527,529,584,586,588,590,592,594,595,596,597,599,600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,619,620,621,622,623,624,625,626,628,629,630,631,633,634,635,637,638,639,640,641,642,643,644,646,647,648,650,651,652,653,654,655,656,658,659,660,661,662,663,664,665,666,667,668,669,671,672,673,675,677
#SBATCH --array=155
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
OUTDIR=${BASEDIR}/output_20260114
#SLURM_ARRAY_TASK_ID=10

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
				 --seg-kidney-t1-mdr-fix=${BASEDIR}/SEG_KIDNEY_T1_FIXES \
				 --t1-scale-factors=${BASEDIR}/afirm_t1_scale_factors_newids.csv \
				 --autoskip  --force=dwi_moco 


#--force seg_kidney_t2w_nocyst,t1_raw_smoothed,t1_molli_mdr,t1_molli_nomdr

				 #--skip=pcasl_moco,fair_moco --force organ_dixon_radiomics
				 #--autoskip --force seg_kidney_whole,seg_kidney_pelvis_trace,seg_kidney_pelvis_trace_splitlr,wkv_ero_radiomics,wkv_ero_shape_metrics,wkv_radiomics,wkv_shape_metrics
				 # --force=temp_add_pelvis,seg_kidney_t2w_fix,seg_kidney_cyst_t2w_trace,seg_kidney_cyst_t2w_trace_splitlr,kidney_cyst_stats_trace_fixed,kidney_cyst_stats_trace_orig,cyst_volume_diffs,seg_organs_trace,seg_kidney_pelvis_trace,seg_kidney_pelvis_trace_splitlr,wkv_ero_radiomics,wkv_ero_shape_metrics,wkv_radiomics,wkv_shape_metrics \
				 #--skipdone=t1_molli,t1_molli_mdr,t1_molli_nomdr,t1_se_mdr,t1_se_nomdr,t1_se_mdr_step2,t2,t2star,mtr,b0,b1,dwi_moco,dwi_adc,pcasl_moco,fair_moco,seg_kidney_t1,seg_kidney_t1_mdr,seg_kidney_t1_nomdr,seg_kidney_t1_se,seg_kidney_cyst_t2w,seg_sat_dixon_cor,seg_liver_dixon_cor,seg_spleen_dixon_cor,seg_kidney_dixon_cor,seg_sat_dixon_ax,seg_liver_dixon_ax,seg_spleen_dixon_ax,seg_kidney_dixon_ax,seg_kidney_t2w,totalseg
