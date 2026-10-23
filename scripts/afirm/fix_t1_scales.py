BASE = "/spmstore/project/RenalMRI/afirm"
INFILE = f"{BASE}/afirm_t1_scale_factors.csv"
SUBFILE = f"{BASE}/afirm_subjects_summary_20251017.csv"
OUTFILE = f"{BASE}/afirm_t1_scale_factors_fixed.csv"

import csv

# Read subjects summary as list of lists
with open(SUBFILE, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    subjects = [row for row in reader]

# Read CSV as list of dicts
with open(INFILE, newline="") as f:
    reader = csv.DictReader(f)
    scales = [row for row in reader]

found = []
print("subjid,SCALE FACTOR LEFT,SCALE FACTOR RIGHT")
for row in subjects:
    subjid, expid, xnat, site, idx, visit, baseline, reason = tuple(row)
    scale_rows = [
        r for r in scales if r["\ufeffsubjid"].upper() == f"{subjid}_{expid}".upper()
    ]
    if not scale_rows:
        #print(f" - No scale factor found for {subjid}_{expid}")
        continue
    elif len(scale_rows) > 1:
        print(f" - Multiple scale factors found for {subjid}_{expid}")
        continue
    scale_row = scale_rows[0]
    old_subjid = scale_row["\ufeffsubjid"]
    new_subjid = f"{site}_{int(idx):03d}_V{visit}"
    if old_subjid in found:
        print(f" - Duplicate scale factor for {old_subjid}")
        continue
    found.append(old_subjid)
    #print(scale_row)
    print(f"{new_subjid},{scale_row['SCALE FACTOR LEFT']},{scale_row['SCALE FACTOR RIGHT']}")

for row in scales:
    subjid = row["\ufeffsubjid"]
    if subjid not in found:
        print(f"NOMATCH {subjid},{row['SCALE FACTOR LEFT']},{row['SCALE FACTOR RIGHT']}")

