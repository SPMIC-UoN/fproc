DATESTAMP_IN=20240904
DATESTAMP_OUT=20241008

fproc-combine --input /spmstore/project/RenalMRI/tk21/proc_${DATESTAMP_IN}/ --output /spmstore/project/RenalMRI/tk21/tk21_${DATESTAMP_OUT}.csv --path fproc/stats/stats.csv tkv_fix_out/tkv.csv shape_metrics_out/tkv_shape_metrics.csv --path-csv seg_stats_out/stats.csv

fproc-flatten --input  /gpfs01/spmstore/project/RenalMRI/tk21/proc_${DATESTAMP_IN} --output /gpfs01/spmstore/project/RenalMRI/tk21/imgs_${DATESTAMP_OUT}

