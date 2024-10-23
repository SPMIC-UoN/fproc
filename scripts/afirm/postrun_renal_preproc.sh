DATEIN=20240807
DATEOUT=20241023

fproc-combine --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/afirm_idps_${DATEOUT}.csv --path-csv seg_stats_out/stats.csv --path tkv_fix_out/tkv.csv volumes_out/tkv_vols.csv fproc_t2/stats/stats.csv shape_metrics_out/tkv_shape_metrics.csv fproc_t2/seg_kidney_cyst_t2w/kidney_cyst.csv
fproc-flatten --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/imgs_${DATEOUT}

