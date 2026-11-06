# IMPORTS *****************************************************************************************
# External libraries
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

# Constants ***************************************************************************************
# The seven BRACS WSI classes, in clinical order from benign to malignant
BRACS_CLASSES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# Helper functions ********************************************************************************
def _read_rows(labels_file: str | Path, sheet: int) -> list[dict[str, str]]:
    """Read one worksheet of an .xlsx file (a zip of XML files) as a list of {column letter: text} rows."""

    # STEP 1: Read the shared strings and the worksheet
    with zipfile.ZipFile(labels_file) as z:
        strings = ["".join(t.text or "" for t in si.iter(_NS + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        root = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet}.xml"))

    # STEP 2: Turn every non-empty row into {column letter: text}
    rows = []
    for row in root.iter(_NS + "row"):
        cells = {}
        for c in row.findall(_NS + "c"):
            v = c.find(_NS + "v")
            if v is not None:
                cells[re.match(r"[A-Z]+", c.get("r")).group()] = strings[int(v.text)] if c.get("t") == "s" else v.text
        if cells:
            rows.append(cells)

    return rows


# Main functions **********************************************************************************
def load_bracs_labels(labels_file: str | Path) -> dict[str, dict]:
    """
    Read the WSI labels from the BRACS spreadsheet (first sheet), without extra dependencies.

    The sheet needs the columns 'WSI Filename', 'Patient Id', 'WSI label' and 'Set'.

    * Arguments:
    labels_file: The path to BRACS.xlsx

    Returns {slide_stem: {"label": str, "patient": str, "set": str}}, e.g. "BRACS_264" -> IC / 30 / Training.
    """

    # STEP 1: Read the rows and find the columns by header name
    rows = _read_rows(labels_file, 1)
    header = {text.strip(): col for col, text in rows[0].items()}
    missing = {"WSI Filename", "Patient Id", "WSI label", "Set"} - set(header)
    if missing:
        raise ValueError(f"Columns not found in {labels_file}: {sorted(missing)}")

    # STEP 2: Build the lookup
    return {
        r[header["WSI Filename"]].strip(): {
            "label": r[header["WSI label"]].strip(),
            "patient": r[header["Patient Id"]].strip(),
            "set": r.get(header["Set"], "").strip(),
        }
        for r in rows[1:]
        if header["WSI Filename"] in r and header["WSI label"] in r and header["Patient Id"] in r
    }


def load_bracs_roi_distribution(labels_file: str | Path) -> dict[str, dict[str, int]]:
    """
    Read the number of regions of interest (RoIs) per slide and per RoI class (second sheet).

    Only slides that have an RoI breakdown in the sheet are returned (not necessarily all slides).

    * Arguments:
    labels_file: The path to BRACS.xlsx

    Returns {slide_stem: {"total": int, "N": int, "PB": int, ...}}; missing counts ("-") are 0.
    """

    # STEP 1: Find the header row ('WSI Filename', ..., 'Total RoI number', 'N', 'PB', ...)
    rows = _read_rows(labels_file, 2)
    head_index = next(i for i, r in enumerate(rows) if "WSI Filename" in {v.strip() for v in r.values()})
    header = {text.strip(): col for col, text in rows[head_index].items()}

    # STEP 2: Build the lookup, turning '-' into 0
    count = lambda r, name: int(r.get(header[name], "0").strip().replace("-", "0") or 0)
    return {
        r[header["WSI Filename"]].strip(): {"total": count(r, "Total RoI number"), **{c: count(r, c) for c in BRACS_CLASSES}}
        for r in rows[head_index + 1:]
        if header["WSI Filename"] in r
    }
