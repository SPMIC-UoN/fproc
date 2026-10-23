DATEIN=20250130
DATEOUT=20250421
STUDY=afirm

fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_kidney_${DATEOUT}.csv \
	--path fproc/stats/stats.csv \
	       fproc/t1_molli_md/t1_molli_md.csv 

fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
        --output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_t1_radiomics_left_${DATEOUT}.csv \
	--path fproc/t1_radiomics_left/radiomics.csv
	
fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_t1_radiomics_right_${DATEOUT}.csv \
	--path fproc/t1_radiomics_right/radiomics.csv

fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_organs_${DATEOUT}.csv \
	--path fproc/stats_dixon/stats.csv \
	   	   fproc/kidney_dixon_shape_metrics_ax/shape_metrics.csv \
		   fproc/kidney_dixon_shape_metrics_cor/shape_metrics.csv \
	   	   fproc/kidney_dixon_radiomics_ax/radiomics.csv \
		   fproc/kidney_dixon_radiomics_cor/radiomics.csv

fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_shape_${DATEOUT}.csv \
	--path fproc/seg_kidney_t2w_vols/volumes.csv \
	       fproc/tkv_shape_metrics/shape_metrics.csv \
	       fproc/tkv_radiomics/radiomics.csv \
	       fproc/seg_kidney_cyst_t2w_clean/kidney_cyst.csv

# Baseline cohort

fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_baseline_kidney_${DATEOUT}.csv \
	--subjids /gpfs01/spmstore/project/RenalMRI/${STUDY}/baseline_cohort.csv \
	--subjids-col 2 \
	--path fproc/stats/stats.csv \
	       fproc/t1_molli_md/t1_molli_md.csv 

fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_baseline_t1_radiomics_left_${DATEOUT}.csv \
	--subjids /gpfs01/spmstore/project/RenalMRI/${STUDY}/baseline_cohort.csv \
	--subjids-col 2 \
	--path fproc/t1_radiomics_left/radiomics.csv
	
fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_baseline_t1_radiomics_right_${DATEOUT}.csv \
	--subjids /gpfs01/spmstore/project/RenalMRI/${STUDY}/baseline_cohort.csv \
	--subjids-col 2 \
	--path fproc/t1_radiomics_right/radiomics.csv

fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_baseline_organs_${DATEOUT}.csv \
	--subjids /gpfs01/spmstore/project/RenalMRI/${STUDY}/baseline_cohort.csv \
	--subjids-col 2 \
	--path fproc/stats_dixon/stats.csv \
	   	   fproc/kidney_dixon_shape_metrics_ax/shape_metrics.csv \
		   fproc/kidney_dixon_shape_metrics_cor/shape_metrics.csv \
	   	   fproc/kidney_dixon_radiomics_ax/radiomics.csv \
		   fproc/kidney_dixon_radiomics_cor/radiomics.csv

fproc-combine \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/${STUDY}_baseline_shape_${DATEOUT}.csv \
	--subjids /gpfs01/spmstore/project/RenalMRI/${STUDY}/baseline_cohort.csv \
	--subjids-col 2 \
	--path fproc/seg_kidney_t2w_vols/volumes.csv \
	       fproc/tkv_shape_metrics/shape_metrics.csv \
	       fproc/tkv_radiomics/radiomics.csv \
	       fproc/seg_kidney_cyst_t2w_clean/kidney_cyst.csv

fproc-flatten \
	--input  /gpfs01/spmstore/project/RenalMRI/${STUDY}/output_${DATEIN} \
	--output /gpfs01/spmstore/project/RenalMRI/${STUDY}/imgs_${DATEOUT}
