#!/bin/bash

module load ants
module load freesurfer
module load fsl

OUTDIR=/home/bbzmsc/data/daniel_c/healthy_output

for SUBJIDX in {1..7}; do
  fproc --pipeline pipelines/csi.py --input ${OUTDIR} --output ${OUTDIR} --overwrite \
                                    --subjidx ${SUBJIDX} --skipdone="*" --noskip=csi_align
done
