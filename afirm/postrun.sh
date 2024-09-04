DATEIN=20240730
DATEOUT=20240801

fproc-combine --input  /spmstore/project/RenalMRI/afirm/proc_${DATEIN} --output /spmstore/project/RenalMRI/afirm/afirm_fproc_${DATEOUT}.csv --path preproc/stats/stats.csv preproc/cmd/cmd.csv
fproc-flatten --input  /spmstore/project/RenalMRI/afirm/proc_${DATEIN} --output /spmstore/project/RenalMRI/afirm/imgs_fproc_${DATEOUT}

