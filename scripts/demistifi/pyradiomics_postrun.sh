#!/bin/sh

fproc-combine \
        --input /spmstore/project/RenalMRI/demistifi/pyradiomics_output/ \
        --output /spmstore/project/RenalMRI/demistifi/demistifi_pyradiomics_20240715.csv \
        --path pyradiomics/radiomics_features.csv 

