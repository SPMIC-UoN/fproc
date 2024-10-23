#!/bin/sh

MODELDIR=/software/imaging/body_pipelines/trained_models/
INDIR=/spmstore/project/RenalMRI/tk21/proc_20240904
OUTDIR=/spmstore/project/RenalMRI/tk21/proc_20240904

for SUBJDIR in ${INDIR}/*; do
    SUBJID=`basename $SUBJDIR`
    echo "Processing ${SUBJID} in ${SUBJDIR}"
    fproc --pipeline pipelines/tk21_t2.py --input ${OUTDIR} --input-subfolder fsort --output ${OUTDIR} --output-subfolder t2map --overwrite \
                                     --subjid ${SUBJID}
done
