DATESTAMP_IN=20240722
DATESTAMP_OUT=20240812

fproc-combine --input /spmstore/project/RenalMRI/bariatric/proc_${DATESTAMP_IN} --output /spmstore/project/RenalMRI/bariatric/stats_${DATESTAMP_OUT}.csv --path fproc/stats/stats.csv
fproc-flatten --input  /spmstore/project/RenalMRI/bariatric/proc_${DATESTAMP_IN}/ --output /spmstore/project/RenalMRI/bariatric/imgs_${DATESTAMP_OUT}

