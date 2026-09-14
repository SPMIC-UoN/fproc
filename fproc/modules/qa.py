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
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

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
        # Collect per-image results for CSV and per-subject PDFs
        results_list = []
        # Prefer an explicit subject id from pipeline options if available
        subjid = self.pipeline.options.subjid

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
                        test_fail_text = ""
                        if not result:
                            test_fail_text = f"Missing image: {check.image_types}"
                            fail_text += test_fail_text + " / "
                        f.write(f"{ ' '.join(check.image_types) },,Exists,{result_str},{test_fail_text}\n")
                        # Record exists check as a result entry (one entry per check)
                        results_list.append({
                            "fname": " ".join(check.image_types),
                            "vendor": vendor,
                            "check": "Exists",
                            "result": result_str,
                            "fail_text": test_fail_text,
                        })
                        continue
                    elif not imgs:
                        LOG.debug(f"No images found for {vendor} {check.image_types}")
                        continue

                    for img in imgs:
                        LOG.debug(f" - Checking {img.fname}")
                        check_text, result, reason = self.check_image(img, check)
                        result_str = "PASS" if result else "FAIL"
                        overall = overall and result
                        test_fail_text = ""
                        if not result:
                            test_fail_text = img.fname + " " + check_text.replace(",", " ")
                            if reason:
                                test_fail_text += f'[{reason.replace(",", " ")}]'
                            fail_text += test_fail_text + " / "
                        f.write(f"{img.fname},{vendor},{check_text},{result_str},{test_fail_text}\n")
                        results_list.append({
                            "fname": img.fname,
                            "vendor": vendor,
                            "check": check_text,
                            "result": result_str,
                            "fail_text": test_fail_text,
                        })

            f.write(f"ALL FILES,,ALL CHECKS,{'PASS' if overall else 'FAIL'},\n")

        with open(self.outfile("qa.csv"), "w") as f:
            f.write(f"result,{'PASS' if overall else 'FAIL'}\n")
            f.write(f"fail_reasons,{fail_text.rstrip(' / ')}\n")

        # Generate per-subject PDF reports if reportlab is available
        if REPORTLAB_AVAILABLE:
            self._generate_subject_pdfs(results_list, subject=subjid)
        else:
            LOG.info("ReportLab not available - skipping PDF report generation")


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

    def _generate_subject_pdfs(self, results_list, subject=None):
        """Generate a single PDF report for the given subject.

        The pipeline always operates on a single subject (`subjid`), so this
        function writes one PDF using the provided `subject` as header and
        the entire `results_list` as the table contents.
        """
        if not results_list:
            LOG.info("No per-image results to include in PDFs")
            return

        subject_id = subject or "unknown"
        styles = getSampleStyleSheet()

        pdf_path = self.outfile(f"{subject_id}_qa_report.pdf")
        # Reduce margins to fit more content horizontally
        left_margin = right_margin = 18
        top_margin = bottom_margin = 18
        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4, leftMargin=left_margin, rightMargin=right_margin, topMargin=top_margin, bottomMargin=bottom_margin
        )
        elems = []

        title = styles["Title"]
        normal = styles["Normal"]
        elems.append(Paragraph(f"QA Report: {subject_id}", title))
        elems.append(Spacer(1, 6))
        elems.append(Paragraph(f"Generated: {datetime.datetime.now().isoformat()}", normal))
        elems.append(Spacer(1, 12))

        # Table header
        table_data = [["File", "Vendor", "Check", "Result", "Notes"]]

        # Create a smaller paragraph style for wrapped cells
        styles = getSampleStyleSheet()
        small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, leading=10)

        for rec in results_list:
            table_data.append([
                Paragraph(str(rec.get("fname", "")), small),
                Paragraph(str(rec.get("vendor", "")), small),
                Paragraph(str(rec.get("check", "")), small),
                Paragraph(str(rec.get("result", "")), small),
                Paragraph(str(rec.get("fail_text", "")), small),
            ])

        # Compute column widths to fit A4 minus margins
        page_width = A4[0]
        usable_width = page_width - (left_margin + right_margin)
        # Proportions: File 25%, Vendor 12%, Check 38%, Result 8%, Notes 17%
        col_proportions = [0.25, 0.12, 0.38, 0.08, 0.17]
        colWidths = [usable_width * p for p in col_proportions]

        table = Table(table_data, colWidths=colWidths, repeatRows=1, hAlign="LEFT")
        # Style: header background, grid, and highlight FAIL rows
        tbl_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9d9d9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]

        # Apply background for fail rows using the original results_list values
        for i, rec in enumerate(results_list, start=1):
            res = str(rec.get("result", "")).upper()
            if res != "PASS":
                tbl_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#ffe6e6")))

        table.setStyle(TableStyle(tbl_style))
        elems.append(table)

        try:
            doc.build(elems)
            LOG.info(f"Wrote PDF report: {pdf_path}")
        except Exception as e:
            LOG.exception(f"Failed to write PDF report for {subject_id}: {e}")
