DATESTAMP_IN=20240910
DATESTAMP_OUT=20240924

fproc-combine --input /spmstore/project/RenalMRI/mlrepeat/proc_${DATESTAMP_IN} --output /spmstore/project/RenalMRI/mlrepeat/mlrepeat_${DATESTAMP_OUT}.csv --path fproc/stats/stats.csv fproc/radiomics/radiomics.csv
fproc-flatten --input /spmstore/project/RenalMRI/mlrepeat/proc_${DATESTAMP_IN} --output /spmstore/project/RenalMRI/mlrepeat/imgs_${DATESTAMP_OUT}

