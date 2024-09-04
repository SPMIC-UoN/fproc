fproc-combine --input  /spmstore/project/RenalMRI/recover/output_20240311/ --output /spmstore/project/RenalMRI/recover/recover_20240311.csv --path stats/stats.csv shape_metrics_out/tkv_shape_metrics.csv --path-csv seg_stats_out/stats.csv  --skip-empty
fproc-flatten --input  /spmstore/project/RenalMRI/recover/output_20240311/ --output /spmstore/project/RenalMRI/recover/imgs_20240311

