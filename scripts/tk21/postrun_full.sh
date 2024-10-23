DATESTAMP_IN=20240904
DATESTAMP_OUT=20241018

fproc-combine --input /spmstore/project/RenalMRI/tk21/proc_${DATESTAMP_IN} --output /spmstore/project/RenalMRI/tk21/tk21_full_${DATESTAMP_OUT}.csv --path fproc_full/stats/stats.csv fproc_full/cmd/cmd.csv

