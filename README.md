# File pre-sorter for imaging processing pipelines

A common feature of most imaging processing pipelines is an initial step where
input data (in DICOM or NIFTI) format is identified for particular uses in 
the pipelines. For example, we may be expecting to find a T1 map, a T2* weighted
scan and a set of DWI volumes. Usually this identification involves an ad-hoc
mixture of filename matching and metadata inspection.

Fsort is intended to be a configurable drop-in process capable of performing
these kind of matching tasks and writing output files to a folder with
standardized naming. However the semantics of the matching - how we identify
the T1 map - are part of the configuration and need to be tailored for each
application.

An important principle of Fsort is that filenames should never used in matching.
DCM2NIIX creates filenames based on metadata and the metadata should be inspected
directly as that is more reliable than parsing filenames with globs and regular
expressions. However some basic filename matching functionality is available
for use cases where, for example, only NIFTI data is availabel and no JSON
metadata sidecar is provided.
