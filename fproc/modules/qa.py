"""
QA: Configuration driven QC of data
"""
import argparse
from collections import OrderedDict
from email.mime import image
import types
import datetime
import logging
import os
import sys
import re

from fproc.module import Module

LOG = logging.getLogger(__name__)

import pandas as pd
import pydicom
import xnat_nott

FLOAT_TOLERANCE = 0.001
KNOWN_VENDORS = ["philips", "siemens", "ge"]
IGNORE_SCAN = 0

class QA(Module):
    def __init__(self, name="qa", **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        # Read Excel configuration file
        config_file = self.kwargs.get("config", None)
        if not config_file:
            self.no_data("No configuration file specified")
        vendor_checks = self.load_config(config_file)
    
        # Run the checks
        fname = self.outfile("qa_results.csv")
        overall = True
        fail_text = ""
        with open(fname, "w") as f:
            for vendor, checks in vendor_checks.items():
                LOG.info(f"Vendor: {vendor}")
                for check in checks:
                    LOG.debug(str(check))
                    imgs, _missing = self.get_images(vendor, check.image_types, check.ifparam, check.ifvalue)
                    if check.param == "exists":
                        result = len(imgs) > 0
                        result_str = "PASS" if result else "FAIL"
                        overall = overall and result
                        if not result:
                            test_fail_text += f"Missing image: {check.image_types}"
                            fail_text += test_fail_text + " / "
                        else:
                            test_fail_text = ""
                        f.write(f"{' '.join(check.image_types)},,Exists,{result_str},{test_fail_text}\n")
                        continue
                    elif not imgs:
                        LOG.debug(f"No images found for {vendor} {check.image_types}")
                        continue

                    for img in imgs:
                        LOG.debug(f" - Checking {img.fname}")
                        check_text, result, reason = self.check_image(img, check)
                        result_str = "PASS" if result else "FAIL"
                        overall = overall and result
                        if not result:
                            test_fail_text = img.fname + " " + check_text.replace(",", " ")
                            if reason:
                                test_fail_text += f'[{reason.replace(",", " ")}]'
                            fail_text += test_fail_text + " / "
                        else:
                            test_fail_text = ""
                        f.write(f"{img.fname},{vendor},{check_text},{result_str},{test_fail_text}\n")
            
            f.write(f"ALL FILES,,ALL CHECKS,{'PASS' if overall else 'FAIL'},\n")

        with open(self.outfile("qa.csv"), "w") as f:
            f.write(f"result,{'PASS' if overall else 'FAIL'}\n")
            f.write(f"fail_reasons,{fail_text.rstrip(' / ')}\n")


    def load_config(self, config_file, sheet="Sheet1"):
        sheet = pd.read_excel(config_file, dtype=str)

        # Read series name mapping
        #
        # There is a configuration for each vendor. The configuration
        # is a dictionary from substring series description matcher
        # to 'standard' name. The checks worksheet is then specified
        # in terms of 'standard' names to get around different naming
        # conventions for different vendors
        config = sheet.fillna('')
        vendor_checks = {}

        LOG.debug(config)
        cols = [colname.strip().lower() for colname in config.columns]
        LOG.info("Columns:")
        for col_idx, col_name in enumerate(cols):
            LOG.info(f"  - {col_idx}: {col_name}")

        for _index, row in config.iterrows():
            if "vendor" in cols:
                vendors = [v.strip().lower() for v in row[cols.index("vendor")].split(",")]
            else:
                vendors = [""]

            mapping = types.SimpleNamespace()

            # image types
            image_col_idx = cols.index("image") if "image" in cols else None
            if image_col_idx is not None:
                image_types = [v.strip().lower() for v in row[image_col_idx].split(",")]
                mapping.image_types = image_types

            for idx, col_name in enumerate(cols):
                if col_name in ("image", "vendor"):
                    continue
                setattr(mapping, col_name, row[idx].strip().lower())

            for vendor in vendors:
                if vendor not in vendor_checks:
                    vendor_checks[vendor] = []
                vendor_checks[vendor].append(mapping)

        return vendor_checks

    def get_images(self, vendor, image_types, ifparam, ifvalue):
        """
        Get images for a given vendor and image type
        """
        imgs = []
        missing = []
        for image in image_types:
            if "/" in image:
                dir, name = image.split("/", 1)
            else:
                dir, name = image, image
            matches = self.inimgs(dir, f"{name}.nii.gz", src=self.INPUT)
            if not matches:
                missing.append((dir, name))
            if vendor:
                for img in imgs:
                    if not img.vendor:
                        LOG.warning(f" - Image {img.fname} has no vendor information - skipping vendor specific checks")
                matches = [
                    img for img in matches 
                    if img.vendor and img.vendor.lower() == vendor.lower()
                ]
            if ifparam:
                LOG.debug(" - Filtering images by %s(%s) = %s" % (ifparam, [getattr(img, ifparam.strip().lower(), None) for img in matches], ifvalue))
                for img in matches:
                    imgval = str(getattr(img, ifparam.strip().lower(), None))
                    LOG.debug(f"   - {img.fname}: {ifparam.strip().lower()} = '{imgval}:{type(imgval)}' == '{str(ifvalue)}'")
                    LOG.debug(f"   - {imgval == str(ifvalue)}")
                matches = [
                    img for img in matches 
                    if str(getattr(img, ifparam.strip().lower(), None)) == str(ifvalue)
                ]
            imgs.extend(matches)

        return imgs, missing

    def check_image(self, img, check):
        """
        """
        imgval = getattr(img, check.param.strip().lower(), "")
        if check.param.lower() == "echotime":
            try:
                imgval = str(float(imgval) * 1000) # convert to ms
            except ValueError:
                pass

        if check.index:
            check_text = f"{check.param}[{check.index}]: {imgval} {check.operator} {check.expected}"
        else:
            check_text = f"{check.param}: {imgval} {check.operator} {check.expected}"

        if check.ideal:
            check_text += f" (ideal: {check.ideal})"

        try:
            if check.index:
                imgval = str(imgval[int(check.index)])
                LOG.debug(f" - Checking index {check.index} : {imgval}")
            else:
                imgval = str(imgval)
                LOG.debug(f" - Checking {imgval}")

            if check.operator == "range":
                result = self._check_range(check.param, imgval, check.expected)
            elif check.operator == "contain":
                result = check.expected in imgval
            elif check.operator == "does not contain":
                result = check.expected not in imgval
            elif check.operator == "==":
                result = self._check_equality(check.param, imgval, check.expected)
            else:
                raise RuntimeError(f"Unsupported operator for {check.param}: {check.operator}")

            LOG.info(" - %s: %s" % (check_text, "PASS" if result else "FAIL"))
            return check_text, result, ""
        except Exception as e:
            LOG.exception(f"Check failed for {check.param}: {e}")
            return check_text, False, str(e)


    def _check_range(self, param, imgval, expected):
        expected_str = [v for v in expected.strip("[]").split(",")]
        if len(expected_str) != 2:
            raise RuntimeError(f"Invalid range specification for {param}: {expected}")
        expected_num = [float(v) for v in expected_str]
        num_sf = max([self._expected_num_sf(v) for v in expected_str])

        # Allow equality if matches within significant figures
        imgval = float(imgval)
        result = (imgval >= expected_num[0] and imgval <= expected_num[1]) or self._float_matches(imgval, expected_num[0], num_sf) or self._float_matches(imgval, expected_num[1], num_sf)
        return result


    def _check_equality(self, param, imgval, expected):
        num_sf = self._expected_num_sf(expected)
        try:
            return self._float_matches(float(imgval), float(expected), num_sf)
        except ValueError:
            return imgval.strip().lower() == expected.strip().lower()


    def _expected_num_sf(self, floatstr):
        """
        :return: Number of significant figures specified by a floating point
                number in a string, e.g. '1.43' would return 3
        """
        return len(floatstr.strip().replace(".", "").replace("-", "").lstrip("0"))


    def _float_matches(self, f1, f2, num_sf):
        """
        :return: True if f1 and f2 are the same to within num_sf significant figures
        """
        f1 = float(('%.' + str(num_sf) + 'g') % f1)
        f2 = float(('%.' + str(num_sf) + 'g') % f2)
        return abs(f1 - f2) < FLOAT_TOLERANCE
