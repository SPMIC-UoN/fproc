DATESTAMP_IN=20241011
DATESTAMP_OUT=20241014

fproc-combine --input /spmstore/project/RenalMRI/mollihr/output_${DATESTAMP_IN} --output /spmstore/project/RenalMRI/mollihr/mollihr_${DATESTAMP_OUT}.csv --path fproc/stats/stats.csv 
fproc-flatten --input /spmstore/project/RenalMRI/mollihr/output_${DATESTAMP_IN} --output /spmstore/project/RenalMRI/mollihr/imgs_${DATESTAMP_OUT}

