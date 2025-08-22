import logging
import os

import numpy as np

from fproc.module import Module
from fproc.modules import segmentations, seg_postprocess, statistics, maps

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

class Radiomics(statistics.Radiomics):
    def __init__(self):
        statistics.Radiomics.__init__(
            self,
            params={
                "t1_molli" : {"dir" : "t1_molli", "fname" : "t1_conf.nii.gz", "minval" : 200, "maxval" : 1400},
                "t1_se" : {"dir" : "t1_se", "fname" : "t1.nii.gz", "minval" : 200, "maxval" : 1400},
            },
            segs = {
                "liver" : {"dir" : "seg_liver_dixon_fix", "fname" : "liver.nii.gz"},
            }
        )

class SegStats(statistics.SegStats):
    def __init__(self):
        statistics.SegStats.__init__(
            self, name="stats",
            default_limits="3t",
            segs={
                "liver" : {
                    "dir" : "seg_liver_dixon_fix",
                    "glob" : "liver.nii.gz"
                },
                "spleen" : {
                    "dir" : "seg_spleen_dixon",
                    "glob" : "spleen.nii.gz"
                },
                "kidney" : {
                    "dir" : "seg_kidney_dixon",
                    "glob" : "kidney.nii.gz"
                },
                "sat" : {
                    "dir" : "seg_sat_dixon",
                    "glob" : "sat.nii.gz",
                    "params" : [],
                },
            },
            params={
                "t2star" : {
                    "dir" : "t2star_dixon",
                    "glob" : "t2star_exclude_fill.nii.gz",
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
    DixonClassify(dixon_src="../raw_dixon"),
    # Segmentations
    segmentations.BodyDixon(dixon_dir="fproc/dixon_classify"),
    segmentations.SatDixon(dixon_dir="fproc/dixon_classify"),
    segmentations.LiverDixon(dixon_dir="fproc/dixon_classify"),
    segmentations.SpleenDixon(dixon_dir="fproc/dixon_classify"),
    segmentations.KidneyDixon(dixon_dir="fproc/dixon_classify", model_id="422"),
    segmentations.PancreasEthrive(),
    segmentations.KidneyT2w(),
    segmentations.TotalSeg(src_dir="fproc/dixon_classify", dilate=1),

    # Parameter maps
    maps.FatFractionDixon(dixon_dir="fproc/dixon_classify"),
    maps.T2starDixon(dixon_dir="fproc/dixon_classify"),
    T1Molli(),
    T1SE(),
    maps.MTR(),
    maps.T2(),

    # Post-processing of segmentations
    seg_postprocess.SegFix(
        "seg_liver_dixon",
        fix_dir_option="liver_masks",
        segs={
            "liver.nii.gz" : "%s_*.nii.gz",
        },
        map_dir="../dixon",
        map_fname="water.nii.gz"
    ),

    # Statistics
    Radiomics(),
    SegStats(),
]

def add_options(parser):
    parser.add_argument("--add-niftis", help="Dir containing additional NIFTI maps")
    parser.add_argument("--liver-masks", help="Directory containing manual liver masks")
