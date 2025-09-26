import os, re
from typing import Dict, Iterable, Callable, Union, List, Optional
import numpy as np

# types of filters we accept per field
Filter = Union[str, Iterable[str], re.Pattern, Callable[[Optional[str]], bool]]

def find_metadata(
        root_folder: str,
        target_filename: str,
        main_pattern: str,
        secondary_pattern: str,
        filters: Optional[Dict[str, Filter]] = None,
        require_all: bool = False,
) -> List[dict]:
    """
    Walks `root_folder`, finds folders that contain `target_filename`, and extracts
    metadata from folder names using regexes with NAMED GROUPS.

    filters:
      - {"w0": "0.1"}                               -> exact match
      - {"eta": {"1e-3","1e-4"}}                    -> any of set
      - {"method": re.compile(r"(sgd|adam)")}       -> regex
      - {"w0": lambda v: v is not None and float(v) < 0.2}  -> callable predicate

    require_all:
      - If True, only keep results where all known fields are present (non-None).
    """
    # Compile your folder-part regexes with NAMED GROUPS
    # Adjust these once; filtering becomes trivial.
    rx_specs = [
        re.compile(main_pattern),
        re.compile(secondary_pattern),
    ]

    results = []
    for dirpath, _, filenames in os.walk(root_folder):
        if target_filename not in filenames:
            continue

        parts = dirpath.split(os.sep)
        meta = {"w0": None, "eta": None, "method": None, "beta": None}

        for part in parts:
            for rx in rx_specs:
                m = rx.match(part)
                if m:
                    # update only groups present in this regex
                    for k, v in m.groupdict().items():
                        if v is not None:
                            meta[k] = v

        if require_all and any(meta[k] is None for k in meta):
            continue

        if filters and not _match_filters(meta, filters):
            continue

        results.append({
            "path": dirpath,  # or os.path.join(dirpath, target_filename)
            **meta
        })

    return results

def _match_filters(meta, flt):
    for key, cond in flt.items():
        val = meta.get(key)
        if callable(cond):
            if not cond(val):
                return False
        elif isinstance(cond, re.Pattern):
            if val is None or cond.fullmatch(val) is None:
                return False
        elif isinstance(cond, (set, list, tuple)):
            if val not in cond:
                return False
        else:
            if val != str(cond):
                return False
    return True


def get_files_from_metadata(meta_data, searched_file):

    paths_current = []
    path_to_check = meta_data["path"]

    current_files = [
        f for f in os.listdir(path_to_check)
        if re.match(f"{searched_file}" + r"_\d+\.npz$", f)
    ]

    for f in current_files:
        full_path = os.path.join(path_to_check, f)
        paths_current.append({"path": full_path})

    return paths_current

