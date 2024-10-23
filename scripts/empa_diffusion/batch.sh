#!/bin/sh

INDIR=/spmstore/project/RenalMRI/empa/diffusion_overlay
OUTDIR=/spmstore/project/RenalMRI/empa/diffusion_overlay
PREPROCDIR=/spmstore/project/RenalMRI/empa/output_flat

for SUBJDIR in ${INDIR}/*; do
    SUBJID=`basename $SUBJDIR`
    echo "Processing ${SUBJID} in ${SUBJDIR}"
    python pipelines/empa_diffusion.py --input ${OUTDIR} --output ${OUTDIR} --overwrite \
                                  --subjid ${SUBJID} --preproc-output ${PREPROCDIR}
done
