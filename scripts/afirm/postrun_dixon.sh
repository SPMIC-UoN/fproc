DATEIN=20240807
DATEOUT=20241122

fproc-combine --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/afirm_dixon_${DATEOUT}.csv --path fproc_t2/stats_dixon/stats.csv
fproc-flatten --input  /gpfs01/spmstore/project/RenalMRI/afirm/output_${DATEIN} --output /gpfs01/spmstore/project/RenalMRI/afirm/imgs_${DATEOUT}

