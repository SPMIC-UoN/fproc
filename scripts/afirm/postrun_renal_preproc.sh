DATEIN=20240807
DATEOUT=20250113

#fproc-combine --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/afirm_${DATEOUT}.csv --path-csv seg_stats_out/stats.csv --path tkv_fix_out/tkv.csv volumes_out/tkv_vols.csv fproc_t2/stats/stats.csv shape_metrics_out/tkv_shape_metrics.csv fproc_t2/seg_kidney_cyst_t2w_clean/kidney_cyst.csv fproc_t2/tkv_shape_metrics/tkv_shape_metrics.csv fproc_t2/t1_molli_md/t1_molli_md.csv

#fproc-combine --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/afirm_radiomics_right_${DATEOUT}.csv --path fproc_t2/t1_radiomics_right/radiomics.csv

#fproc-combine --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/afirm_radiomics_left_${DATEOUT}.csv --path fproc_t2/t1_radiomics_left/radiomics.csv 

fproc-combine --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/afirm_baseline_${DATEOUT}.csv --path-csv seg_stats_out/stats.csv --path tkv_fix_out/tkv.csv volumes_out/tkv_vols.csv fproc_t2/stats/stats.csv shape_metrics_out/tkv_shape_metrics.csv fproc_t2/seg_kidney_cyst_t2w_clean/kidney_cyst.csv fproc_t2/tkv_shape_metrics/tkv_shape_metrics.csv fproc_t2/t1_molli_md/t1_molli_md.csv --subjids /gpfs01/spmstore/project/RenalMRI/afirm/afirm_baseline_cohort.csv --subjids-col=2

fproc-flatten --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/imgs_${DATEOUT}

