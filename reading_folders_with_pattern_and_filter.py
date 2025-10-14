import os
import re
from typing import Dict, Iterable, Callable, Union, List, Optional, Sequence

# -------- types --------
Filter = Union[str, Iterable[str], re.Pattern, Callable[[Optional[str]], bool]]
PatternLike = Union[str, int, float, re.Pattern]
PatternsLike = Union[PatternLike, Sequence[PatternLike]]


# ==============================
# Core helpers (compilation, filters)
# ==============================

def _match_filters(meta: Dict[str, Optional[str]], flt: Dict[str, Filter]) -> bool:
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


def _compile_one(p: PatternLike) -> re.Pattern:
    """
    Accept a regex string, compiled regex, or numeric literal (int/float).
    Numerics are coerced into an exact full-name match.
    """
    if isinstance(p, re.Pattern):
        return p
    if isinstance(p, (int, float)):  # numeric literal → exact full-name match
        return re.compile(rf"^{str(p)}$")
    return re.compile(str(p))         # treat as regex pattern string


def _to_regex_list(main_pattern: PatternsLike,
                   secondary_pattern: Optional[PatternsLike] = None) -> List[re.Pattern]:
    rx: List[re.Pattern] = []
    if isinstance(main_pattern, (list, tuple)):
        rx.extend(_compile_one(p) for p in main_pattern)
    else:
        rx.append(_compile_one(main_pattern))
    if secondary_pattern is not None:
        if isinstance(secondary_pattern, (list, tuple)):
            rx.extend(_compile_one(p) for p in secondary_pattern)
        else:
            rx.append(_compile_one(secondary_pattern))
    return rx


def _expected_fields(rx_specs: List[re.Pattern]) -> List[str]:
    """Union of all named groups across all patterns (for dynamic meta keys)."""
    fields = set()
    for rx in rx_specs:
        fields.update(rx.groupindex.keys())
    return sorted(fields)


# ==============================
# New utility: list FILE paths by pattern in a folder
# ==============================

def find_paths_by_pattern(
    folder: str,
    pattern: PatternsLike,
    *,
    recursive: bool = False,
    fullmatch: bool = True,
) -> List[str]:
    """
    Return file paths under `folder` whose *basename* matches one or more patterns.

    Args:
        folder: Base directory to scan.
        pattern: Regex string/compiled regex/number, or a list/tuple of them.
                 Examples:
                   r"^trial_(\\d+)\\.npz$"
                   r"^python_file_\\d+\\.py$"
                   123  (exact name '123')
        recursive: If True, descend into subdirectories.
        fullmatch: If True, require full-name match (^...$). If False, use .search().

    Returns:
        Sorted list of matching file paths.
    """
    rx_specs = _to_regex_list(pattern)
    matches: List[str] = []

    def _name_matches(name: str) -> bool:
        for rx in rx_specs:
            if fullmatch:
                if rx.fullmatch(name):
                    return True
            else:
                if rx.search(name):
                    return True
        return False

    def _scan_once(dirpath: str):
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            if _name_matches(entry.name):
                                matches.append(entry.path)
                        elif recursive and entry.is_dir(follow_symlinks=False):
                            _scan_once(entry.path)
                    except OSError:
                        # Skip entries we can't stat (e.g., permissions, broken links)
                        continue
        except FileNotFoundError:
            return

    _scan_once(folder)
    matches.sort()
    return matches


# ==============================
# Your original main API (unchanged behavior)
# ==============================

def find_metadata(
    root_folder: str,
    target_filename: str,
    main_pattern: PatternsLike,
    secondary_pattern: Optional[PatternsLike] = None,
    filters: Optional[Dict[str, Filter]] = None,
    require_all: bool = False,
    include_unnamed: bool = False,  # set True if you want to also capture unnamed groups
) -> List[dict]:
    """
    Walk `root_folder`, keep folders that contain `target_filename`, and extract
    metadata from folder path components using regexes. Meta keys are derived
    from the named groups present in your regex pattern(s).
    """
    rx_specs = _to_regex_list(main_pattern, secondary_pattern)
    fields = _expected_fields(rx_specs)  # dynamic meta keys from patterns

    results: List[dict] = []
    for dirpath, _, filenames in os.walk(root_folder):
        if target_filename not in filenames:
            continue

        parts = dirpath.split(os.sep)
        # initialize only with discovered fields (could be empty if patterns have no named groups)
        meta = {k: None for k in fields}

        for part in parts:
            for rx in rx_specs:
                m = rx.match(part)
                if not m:
                    continue

                # update named groups
                gd = m.groupdict()
                if gd:
                    for k, v in gd.items():
                        if v is not None:
                            meta[k] = v

                # optionally capture unnamed groups too
                if include_unnamed and rx.groups > len(rx.groupindex):
                    named_idx = set(rx.groupindex.values())
                    for j in range(1, rx.groups + 1):
                        if j in named_idx:
                            continue
                        meta_key = f"g{j}_p{id(rx)}"  # unique-ish per pattern object
                        meta[meta_key] = m.group(j)

        # If patterns had no named groups and include_unnamed=False, meta could be {}
        if require_all and fields and any(meta[k] is None for k in fields):
            continue

        if filters and not _match_filters(meta, filters):
            continue

        results.append({"path": dirpath, **meta})

    return results


def get_files_from_metadata(meta_data: dict, searched_file: str) -> List[dict]:
    """
    Return all npz files like '<searched_file>_###.npz' in the matched folder.

    Example:
        meta_data = {"path": "/base/exp/w0_0p1__e_1e-3__m_adam"}
        get_files_from_metadata(meta_data, "fourier_rate")
        -> [{"path": "/base/exp/.../fourier_rate_1.npz"}, ...]
    """
    paths_current: List[dict] = []
    path_to_check = meta_data["path"]

    # Escape searched_file to treat it as a literal (in case it has regex metacharacters)
    base = re.escape(searched_file)
    rx = re.compile(rf"^{base}_(\d+)\.npz$")

    try:
        entries = sorted(os.listdir(path_to_check))
    except FileNotFoundError:
        return paths_current

    for fname in entries:
        if rx.fullmatch(fname):
            full_path = os.path.join(path_to_check, fname)
            paths_current.append({"path": full_path})

    return paths_current