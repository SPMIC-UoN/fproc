#!/bin/sh

INDIR=/spmstore/project/RenalMRI/se_epi/data
OUTDIR=/spmstore/project/RenalMRI/se_epi/output
SUBJIDS=/spmstore/project/RenalMRI/se_epi/subjids.txt

for SUBJIDX in {1..24}; do
  SUBJID=`head -${SUBJIDX} ${SUBJIDS} |tail -1`
  echo "Processing subject ${SUBJIDX}: ${SUBJID}"

  fproc --pipeline pipelines/se_epi.py --input ${INDIR} --datadir ${INDIR} --output ${OUTDIR} --overwrite \
                             --subjid ${SUBJID}
done
