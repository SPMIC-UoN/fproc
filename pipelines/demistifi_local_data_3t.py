# fixed organ masks ONLY for T1-SE on Teams MRQUEE


import logging
import os

import numpy as np

from fproc.module import Module
from fproc.modules import segmentations, seg_postprocess, statistics, maps, regrid

LOG = logging.getLogger(__name__)

class T1Molli(Module):
    def __init__(self):
        Module.__init__(self, "t1_molli")

    def process(self):
        add_niftis = self.pipeline.options.add_niftis
        base_subjid = self.pipeline.options.subjid
        while 1:
            t1s = os.path.join(add_niftis, base_subjid)
            t1 = self.single_inimg("molli_t1_map_nifti", "*.nii.gz", src=t1s)
            if t1:
                break
            base_subjid = base_subjid[:base_subjid.rfind("_")]
            if not base_subjid:
                break
        if t1:
            LOG.info(f" - Saving MOLLI T1 map from {t1.fname}")
            map = t1.data[..., 0]
            conf = t1.data[..., 1]
            t1.save_derived(map, self.outfile("t1_map.nii.gz"))
            t1.save_derived(map, self.outfile("t1_conf.nii.gz"))

class T1SE(Module):
    def __init__(self):
        Module.__init__(self, "t1_se")

    def process(self):
        add_niftis = self.pipeline.options.add_niftis
        base_subjid = self.pipeline.options.subjid
        while 1:
            t1s = os.path.join(add_niftis, base_subjid)
            t1 = self.single_inimg("seepi_t1_map_nifti", "*.nii.gz", src=t1s)
            if t1:
                break
            base_subjid = base_subjid[:base_subjid.rfind("_")]
            if not base_subjid:
                break
        if t1:
            LOG.info(f" - Saving SE T1 map from {t1.fname}")
            t1.save(self.outfile("t1.nii.gz"))


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2
import nibabel as nib

class DixonClassify(Module):
    def __init__(self, **kwargs):
        Module.__init__(self, "dixon_classify", **kwargs)

    def process(self):
        model_fpath = self.kwargs.get("model", "/spmstore/project/RenalMRI/dixon_classifier/dixon_classifier.h5")
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(128,128, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
        model.load_weights(model_fpath)

        dixon_src = self.kwargs.get("dixon_src", "raw_dixon")
        imgs = self.inimgs(dixon_src, "raw_dixon*.nii.gz", src=self.OUTPUT)
        classified_imgs = set()
        class_name = {0: "fat_fraction", 1: "t2star", 2: "water", 3: "fat", 4: "ip", 5: "op"}
        img_probs = []
        if not imgs:
            self.no_data(f"No dixon data found in {dixon_src}")
        for img in imgs:
            for vol in range(img.nvols):
                if len(img.shape) > 3:
                    data = img.data[..., vol]
                else:
                    data = img.data
                img_slices = []
                axcodes = nib.orientations.aff2axcodes(img.affine)
                ax_axis = axcodes.index("S") if "S" in axcodes else axcodes.index("I")
                if ax_axis == 0:
                    data_tp = np.transpose(img, (1, 2, 0))
                elif ax_axis == 1:
                    data_tp = np.transpose(img, (0, 2, 1))
                else:
                    data_tp = data
                max, min = np.percentile(data_tp, 99), np.percentile(data_tp, 1)
                for z in range(data_tp.shape[2]):
                    s = data_tp[:, :, z]
                    res = cv2.resize(s, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                    res = (res - min) / (max - min)
                    img_slices.append(res)
                img_slices = np.array(img_slices)
                predict = model.predict(img_slices)
                predict_mean_slices = np.mean(predict, axis=0)
                for idx, name in class_name.items():
                    img_probs.append((img, vol, data, name, predict_mean_slices[idx]))

        img_probs.sort(key=lambda x: x[4], reverse=True)
        for img, vol, data, name, prob in img_probs:
            if name not in classified_imgs:
                LOG.info(f" - {img.fname} vol {vol} classified as {name} ({prob})")
                img.save_derived(data, self.outfile(f"{name}.nii.gz"))
                classified_imgs.add(name)
            else:
                LOG.info(f" - {img.fname} vol {vol} predicted as {name} ({prob}) but already have better")
            if len(classified_imgs) == 6:
                break


__version__ = "0.0.1"

NAME = "mrquee_bsmart_uon"

MODULES = [
    ## Parameter maps
    maps.DixonClassify(dixon_src="../raw_dixon"),
    maps.FatFractionDixon(dixon_dir="fproc/dixon_classify"),
    maps.T2starDixon(dixon_dir="fproc/dixon_classify"),
    T1Molli(),
    T1SE(),

    ## Segmentations
    segmentations.BodyDixon(dixon_dir="fproc/dixon_classify"),
    segmentations.TotalSeg(src_dir="fproc/dixon_classify", dilate=1),
    segmentations.VatDixon(
        name="seg_vat_dixon",
        ff_dir="fat_fraction",
        ff_glob="fat_fraction_scanner.nii.gz",
        body_dir="seg_body_dixon",
        sat_dir="totalseg",
        sat_glob="subcutaneous_fat.nii.gz",
        organs={
            "totalseg": "liver.nii.gz",
            "totalseg": "spleen.nii.gz",
            "totalseg": "pancreas.nii.gz",
            "totalseg": "kidneys.nii.gz",
        },
    ),
    seg_postprocess.SegFix(
        name="totalseg_fix_t1_se",
        seg_dir="totalseg",
        fix_dir_option="totalseg_fixes",
        segs={
             "liver.nii.gz": {
                "glob": "%s/*liver*.nii.gz",
                "fname": "liver.nii.gz",
            },
             "spleen.nii.gz": {
                "glob": "%s/*spleen*.nii.gz",
                "fname": "spleen.nii.gz",
            },
        }
    ),

    ## Statistics
    statistics.Radiomics(
        name="liver_radiomics",
        params={
            "t1_molli" : {"dir" : "t1_molli", "fname" : "t1_conf.nii.gz", "minval" : 500, "maxval" : 1300},
            "t1_se" : {"dir" : "t1_se", "fname" : "t1_map.nii.gz", "minval" : 500, "maxval" : 1300},
        },
        segs = {
            "liver" : {"dir" : "totalseg_fix_t1_se", "fname" : "liver.nii.gz"},
        },
        features={
            "firstorder" : ["90Percentile", "TotalEnergy"],
        },
        image_types=[
            "Original"
        ],
        deps=["t1_molli", "t1_se", "totalseg_fix_t1_se"]
    ),

    statistics.Radiomics(
        name="spleen_radiomics",
        params={
            "t1_molli" : {"dir" : "t1_molli", "fname" : "t1_conf.nii.gz", "minval" : 900, "maxval" : 1660},
            "t1_se" : {"dir" : "t1_se", "fname" : "t1_map.nii.gz", "minval" : 900, "maxval" : 1660},
        },
        segs = {
            "spleen" : {"dir" : "totalseg_fix_t1_se", "fname" : "spleen.nii.gz"},
        },
        features={
            "firstorder" : ["90Percentile", "TotalEnergy"],
        },
        image_types=[
            "Original"
        ],
        deps=["t1_molli", "t1_se", "totalseg_fix_t1_se"]
    ),
    
    statistics.Radiomics(
        name="pancreas_radiomics",
        params={
            "t1_molli" : {"dir" : "t1_molli", "fname" : "t1_conf.nii.gz", "minval" : 400, "maxval" : 1300},
            "t1_se" : {"dir" : "t1_se", "fname" : "t1_map.nii.gz", "minval" : 400, "maxval" : 1300},
        },
        segs = {
            "pancreas" : {"dir" : "totalseg", "fname" : "pancreas.nii.gz"},
        },
        features={
            "firstorder" : ["90Percentile", "TotalEnergy"],
        },
        image_types=[
            "Original"
        ],
        deps=["t1_molli", "t1_se", "totalseg"]
    ),

    statistics.Radiomics(
        name="lung_radiomics",
        params={
            "water_dixon" : {"dir" : "dixon_classify", "fname" : "water.nii.gz"},
        },
        segs = {
            "lung" : {"dir" : "totalseg", "glob" : "*lung*dilated.nii.gz"},
        },  
        features={
            "firstorder" : ["Uniformity"],
            "glcm" : ["Autocorrelation", "DifferenceVariance", "ClusterTendency"],
            "glszm" : ["ZonePercentage", "ZoneEntropy"],
            "glrlm" : ["RunPercentage", "RunEntropy"],
            "ngtdm" : ["Coarseness"],
        },
        image_types=[
            "Original"
        ],
        deps=["water_dixon", "totalseg"]
    ),
    statistics.ShapeMetrics(
        name="shape_metrics",
        seg_dir="totalseg",
        segs={
            "kidney_left" : "kidney_left.nii.gz", 
            "kidney_right" : "kidney_right.nii.gz",
            "liver" : "liver.nii.gz",
            "spleen" : "spleen.nii.gz",
            "pancreas" : "pancreas.nii.gz",
        },
        metrics=[
            "compactness",
            "long_axis",
            "short_axis",
            "mi_mean",
            "fa",
        ],
    ),
    statistics.Radiomics(
        name="organ_radiomics",
        deps=["dixon_classify", "totalseg"],
        params={
            "radiomics" : {"dir" : "dixon_classify", "fname" : "water.nii.gz"},
        },
        segs={
            "kidney_left": {"dir": "totalseg", "fname": "kidney_left.nii.gz"},
            "kidney_right": {"dir": "totalseg", "fname": "kidney_right.nii.gz"},
            "liver": {"dir": "totalseg", "fname": "liver.nii.gz"},
            "spleen": {"dir": "totalseg", "fname": "spleen.nii.gz"},
            "pancreas": {"dir": "totalseg", "fname": "pancreas.nii.gz"},
        },
        features={
            "shape": [
                "SurfaceArea",
                "VoxelVolume",
                "SurfaceVolumeRatio",
                "MajorAxisLength",
                "MinorAxisLength",
                "Elongation",
                "Compactness1",
            ],
        },
    ), 
    statistics.SegStats(
        name="stats",
        default_limits="3t",
        segs={
            "liver" : {
                "dir" : "totalseg",
                "glob" : "liver.nii.gz",
                "params" : ["t2star", "r2star", "ff"]
            },
            "spleen" : {
                "dir" : "totalseg",
                "glob" : "spleen.nii.gz",
                "params" : ["t2star", "r2star", "ff"]
            },
            "liver_fix" : {
                "dir" : "totalseg_fix_t1_se",
                "glob" : "liver.nii.gz",
                "params" : ["t1_se", "t1_molli"]
            },
            "spleen_fix" : {
                "dir" : "totalseg_fix_t1_se",
                "glob" : "spleen.nii.gz",
                "params" : ["t1_se", "t1_molli"]
            },
            "kidney" : {
                "dir" : "totalseg",
                "glob" : "kidneys.nii.gz"
            },
            "kidney_left" : {
                "dir" : "totalseg",
                "glob" : "kidney_left.nii.gz",
                "params" : ["ff"],
            },
            "kidney_right" : {
                "dir" : "totalseg",
                "glob" : "kidney_right.nii.gz",
                "params" : ["ff"],
            },
            "pancreas" : {
                "dir" : "totalseg",
                "glob" : "pancreas.nii.gz",
            },
            "sat" : {
                "dir" : "totalseg",
                "glob" : "subcutaneous_fat.nii.gz",
                "params" : [],
            },
            "vat" : {
                "dir" : "seg_vat_dixon",
                "glob" : "vat.nii.gz",
                "params" : [],
            },
        },
        params={
            "t2star" : {
                "dir" : "t2star_dixon",
                "glob" : "t2star_exclude_fill.nii.gz",
            },
            "r2star" : {
                "dir" : "t2star_dixon",
                "glob" : "r2star_t2star_exclude_fill.nii.gz",
            },
            "ff" : {
                "dir" : "fat_fraction",
                "glob" : "fat_fraction_scanner.nii.gz",
            },
            "t1_molli" : {
                "dir" : "t1_molli",
                "glob" : "t1_conf.nii.gz",
            },
            "t1_se" : {
                "dir" : "t1_se",
                "glob" : "t1.nii.gz",
            },
        },
        stats=["n", "iqn", "iqmean", "median", "iqstd", "mode", "fwhm"],
        seg_volumes=True,
    )
]

def add_options(parser):
    parser.add_argument("--add-niftis", help="Dir containing additional NIFTI maps")
    parser.add_argument("--totalseg-fixes", help="Dir containing totalseg fixes")
