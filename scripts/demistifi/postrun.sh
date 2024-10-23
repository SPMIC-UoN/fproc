#!/bin/sh

OUTDIR=output_newsubjs
#GROUP=HC
#GROUP=Kidney_disease
#GROUP=Liver_disease
#GROUP=Pancreas_disease
#GROUP=Spleen_disease
GROUP=KIDNEY_DISEASE
#GROUP=NO_KIDNEY_DISEASE
#GROUP=NO_KIDNEY_DATA
TIMESTAMP=20240918

fproc-combine \
        --input /imgshare/ukbiobank/demistifi_output_fullsets/${GROUP}/ \
        --output /spmstore/project/RenalMRI/demistifi/demistifi_hr_newsubjs_${GROUP}_${TIMESTAMP}.csv \
        --path fproc/molli_hr/hr_timings.csv 

