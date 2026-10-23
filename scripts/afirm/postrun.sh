#DATEIN=20251017
DATEIN=20260114
DATEOUT=20260119
BASE=/gpfs01/spmstore/project/RenalMRI
STUDY=afirm

mkdir -p ${BASE}/${STUDY}/${DATEIN}

fproc-combine \
 	--input  ${BASE}/${STUDY}/output_${DATEIN} \
 	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_kidney_${DATEOUT}.csv \
 	--path fproc/scan_dates/scan_dates.csv \
	       fproc/stats/stats.csv \
 	       fproc/t1_molli_md/t1_molli_md.csv 

fproc-combine \
 	--input  ${BASE}/${STUDY}/output_${DATEIN} \
         --output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_t1_radiomics_left_${DATEOUT}.csv \
 	--path fproc/scan_dates/scan_dates.csv \
	       fproc/t1_radiomics_left/radiomics.csv

fproc-combine \
 	--input  ${BASE}/${STUDY}/output_${DATEIN} \
 	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_t1_radiomics_right_${DATEOUT}.csv \
 	--path fproc/scan_dates/scan_dates.csv \
	       fproc/t1_radiomics_right/radiomics.csv

fproc-combine \
 	--input  ${BASE}/${STUDY}/output_${DATEIN} \
 	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_organs_${DATEOUT}.csv \
 	--path fproc/scan_dates/scan_dates.csv \
	       fproc/stats_dixon/stats.csv \
 	   	   fproc/kidney_dixon_shape_metrics_ax/shape_metrics.csv \
 		   fproc/kidney_dixon_shape_metrics_cor/shape_metrics.csv \
 	   	   fproc/kidney_dixon_radiomics_ax/radiomics.csv \
 		   fproc/kidney_dixon_radiomics_cor/radiomics.csv

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_shape_${DATEOUT}.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/seg_kidney_t2w_vols/volumes.csv \
	       fproc/tkv_shape_metrics/shape_metrics.csv \
	       fproc/tkv_radiomics/radiomics.csv \
		   fproc/wkv_volumes/stats.csv \
	       fproc/wkv_shape_metrics/shape_metrics.csv \
	       fproc/wkv_radiomics/radiomics.csv \
		   fproc/wkv_ero_volumes/stats.csv \
	       fproc/wkv_ero_shape_metrics/shape_metrics.csv \
	       fproc/wkv_ero_radiomics/radiomics.csv \
	       fproc/kidney_cyst_stats_trace_fixed/kidney_cyst.csv \
		   fproc/kidney_pelvis_stats_trace/stats.csv

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_shape_qa_${DATEOUT}.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/kidney_stats_alternate/stats.csv \
	       fproc/kidney_cyst_stats_trace_fixed/kidney_cyst.csv \
	       fproc/kidney_cyst_stats_trace_orig/kidney_cyst.csv \
	       fproc/kidney_cyst_stats/kidney_cyst.csv \
		   fproc/cyst_volume_diffs/diffs.csv

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_organ_vols_${DATEOUT}.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/organ_volumes/stats.csv 

# Baseline cohort

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_kidney_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/stats/stats.csv \
	       fproc/t1_molli_md/t1_molli_md.csv 

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_t1_radiomics_left_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/t1_radiomics_left/radiomics.csv
	
fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_t1_radiomics_right_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/t1_radiomics_right/radiomics.csv

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_organs_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/stats_dixon_totalseg/stats.csv \
	   	   fproc/kidney_dixon_shape_metrics_totalseg_ax/shape_metrics.csv \
		   fproc/kidney_dixon_shape_metrics_totalseg_cor/shape_metrics.csv \
	   	   fproc/kidney_dixon_radiomics_totalseg_ax/radiomics.csv \
		   fproc/kidney_dixon_radiomics_totalseg_cor/radiomics.csv \
		   fproc/spleen_shape_metrics_totalseg_cor/shape_metrics.csv \
		   fproc/liver_shape_metrics_totalseg_cor/shape_metrics.csv \
		   fproc/pancreas_shape_metrics_totalseg_cor/shape_metrics.csv 

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_organs_local_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/stats_dixon/stats.csv \
	   	   fproc/kidney_dixon_shape_metrics_ax/shape_metrics.csv \
		   fproc/kidney_dixon_shape_metrics_cor/shape_metrics.csv \
	   	   fproc/kidney_dixon_radiomics_ax/radiomics.csv \
		   fproc/kidney_dixon_radiomics_cor/radiomics.csv

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_t2w_radiomics_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/t2w_radiomics/radiomics.csv

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_shape_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/seg_kidney_t2w_vols/volumes.csv \
	       fproc/tkv_shape_metrics/shape_metrics.csv \
	       fproc/tkv_radiomics/radiomics.csv \
		   fproc/wkv_volumes/stats.csv \
	       fproc/wkv_shape_metrics/shape_metrics.csv \
	       fproc/wkv_radiomics/radiomics.csv \
		   fproc/wkv_ero_volumes/stats.csv \
	       fproc/wkv_ero_shape_metrics/shape_metrics.csv \
	       fproc/wkv_ero_radiomics/radiomics.csv \
	       fproc/kidney_cyst_stats_trace_fixed/kidney_cyst.csv \
		   fproc/kidney_pelvis_stats_trace/stats.csv

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_shape_qa_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/kidney_stats_alternate/stats.csv \
	       fproc/kidney_cyst_stats_trace_fixed/kidney_cyst.csv \
	       fproc/kidney_cyst_stats_trace_orig/kidney_cyst.csv \
	       fproc/kidney_cyst_stats/kidney_cyst.csv \
		   fproc/cyst_volume_diffs/diffs.csv

fproc-combine \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/${STUDY}_baseline_organ_vols_${DATEOUT}.csv \
	--subjids ${BASE}/${STUDY}/baseline_cohort_2025.csv \
	--path fproc/scan_dates/scan_dates.csv \
	       fproc/organ_volumes/stats.csv

fproc-flatten \
	--input  ${BASE}/${STUDY}/output_${DATEIN} \
	--output ${BASE}/${STUDY}/${DATEIN}/imgs_${DATEOUT}
