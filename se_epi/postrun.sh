DATESTAMP_OUT=20240813

fproc-combine --input /spmstore/project/RenalMRI/se_epi/output --output /spmstore/project/RenalMRI/se_epi/stats_${DATESTAMP_OUT}.csv --path stats/stats.csv
fproc-flatten --input  /spmstore/project/RenalMRI/se_epi/output/ --output /spmstore/project/RenalMRI/se_epi/imgs_${DATESTAMP_OUT}

