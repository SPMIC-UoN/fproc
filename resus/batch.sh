#!/bin/sh

MODELDIR=/spmstore/project/RenalMRI/trained_models/
INDIR=/spmstore/project/RenalMRI/resus/proc
OUTDIR=/spmstore/project/RenalMRI/resus/proc

module load nnunetv2-img

for SUBJDIR in ${INDIR}/*; do
    SUBJID=`basename $SUBJDIR`
    echo "Processing ${SUBJID} in ${SUBJDIR}"
    python pipelines/resus_proc.py --input ${OUTDIR} --output ${OUTDIR} --overwrite \
                                  --subjid ${SUBJID}
done
