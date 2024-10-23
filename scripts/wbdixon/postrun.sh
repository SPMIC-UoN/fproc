DATESTAMP_IN=20241014
DATESTAMP_OUT=20241017

fproc-combine --input /spmstore/project/RenalMRI/wbdixon/output_${DATESTAMP_IN} --output /spmstore/project/RenalMRI/wbdixon/wbdixon_${DATESTAMP_OUT}.csv --path fproc/stats/stats.csv 
fproc-flatten --input /spmstore/project/RenalMRI/wbdixon/output_${DATESTAMP_IN} --output /spmstore/project/RenalMRI/wbdixon/imgs_${DATESTAMP_OUT}

