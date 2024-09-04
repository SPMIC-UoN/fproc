#!/bin/sh

MODELDIR=/software/imaging/body_pipelines/trained_models/
INDIR=/spmstore/project/RenalMRI/tk21/proc
OUTDIR=/spmstore/project/RenalMRI/tk21/proc

for SUBJDIR in ${INDIR}/*; do
    SUBJID=`basename $SUBJDIR`
    echo "Processing ${SUBJID} in ${SUBJDIR}"
    fproc --pipeline pipelines/tk21_t2.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder t2map --overwrite \
                                     --subjid ${SUBJID}
done
