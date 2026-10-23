import logging

from fproc.module import Module

__version__ = "0.0.1"

LOG = logging.getLogger(__name__)

NAME = "afirm_t1_multicon"
STUDYDIR = "/spmstore/project/RenalMRI/afirm_t1_multicon"
OUTNAME = NAME

BATCH = {
    "modules": [
        "renal-preproc-img",
        "fsl-img/6.0.6.3",
    ],
    "subjects": 500,
    "time": "04:00:00",
    "mem_per_cpu": "16g",
    "output_subfolder" : "seg",
    "extra_fproc_args" : [
        "--t1-model=/software/imaging/body_pipelines/trained_models//kidney_t1_multicon_min_max.pt",
    ],
}

class T1MulticonSeg(Module):
    def __init__(self):
        Module.__init__(self, "t1_seg")

    def process(self):
        t1map = self.inimg("t1multicon", "t1_multicon.nii.gz")
        LOG.info(
            f" - Segmenting multi-contrast T1 map from {t1map.fname} using model {self.pipeline.options.t1_model}"
        )
        self.runcmd(
            [
                "kidney_t1_seg",
                "--input",
                t1map.dirname,
                "--subjid",
                "",
                "--display-id",
                self.pipeline.options.subjid,
                "--t1",
                t1map.fname,
                "--model",
                self.pipeline.options.t1_model,
                "--multicon",
                "--noclean",
                "--output",
                self.outdir,
                "--outprefix",
                f"seg_kidney_multicon",
            ],
            logfile=f"seg_kidney_multicon.log",
        )


MODULES = [
    T1MulticonSeg(),
]

def add_options(parser):
    parser.add_argument(
        "--t1-model",
        help="Filename or URL for T1 segmentation model weights",
        default="",
    )
