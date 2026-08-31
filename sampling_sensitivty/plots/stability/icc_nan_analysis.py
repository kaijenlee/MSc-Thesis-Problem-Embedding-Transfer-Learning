"""
Locate degenerate (NaN) ICC cells in the pickles written by compute_ela_icc.py:

    ela_icc[config_key][(func, dim)][feature_group][feature_name] = icc_value

A "cell" is one (feature) leaf. A cell is counted as NaN if its value is None
or not finite (np.nan / inf) -- i.e. exactly the cells the plotting code drops.

nan_icc_report(source, dimension) returns two DataFrames:

  detail  : one row per (strategy, size, func) -- how many feature cells were
            NaN, out of how many, and which features. (Only rows with >=1 NaN
            by default; set only_nan=False to keep every combination.)
  summary : one row per (strategy, size) -- total NaN cells, fraction, how many
            functions/features were affected, and the worst offending features.
"""

import re
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def split_config_key(config_key):
    """'lhs_random_cd_75' -> ('lhs_random_cd', 75)."""
    m = re.match(r"^(.*)_(\d+)$", config_key)
    if not m:
        raise ValueError(f"Cannot parse strategy/size from config key {config_key!r}")
    return m.group(1), int(m.group(2))


def load_icc_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_source(source, dimension):
    if isinstance(source, dict):
        if any(isinstance(k, str) for k in source):   # already an ela_icc dict
            return source
        if dimension in source:
            return load_icc_pkl(source[dimension])
        raise KeyError(f"dimension {dimension} not in source keys {sorted(source)}")
    return load_icc_pkl(source)


def _is_nan(v):
    if v is None:
        return True
    try:
        return not np.isfinite(v)
    except TypeError:
        return True


def nan_icc_report(source, dimension, strategy=None, only_nan=True,
                   top_n_features=3, verbose=False):
    """
    Parameters
    ----------
    source : {dim: path} | path | already-loaded ela_icc dict
    dimension : int -- selects the pkl (if source is {dim: path}) and filters (func, dim) keys
    strategy : str | None -- restrict to one sampling strategy (default: all)
    only_nan : bool -- detail keeps only (strategy, size, func) rows with >=1 NaN
    top_n_features : int -- how many worst features to list per (strategy, size) in summary
    verbose : bool -- print a one-line headline

    Returns
    -------
    (detail_df, summary_df)
    """
    ela_icc = _resolve_source(source, dimension)

    records = []                       # full per-(strategy,size,func) records
    # feat_counter[(strategy,size)][feature] = # of funcs where that feature was NaN
    feat_counter = defaultdict(Counter)

    for config_key, by_func in ela_icc.items():
        st, size = split_config_key(config_key)
        if strategy is not None and st != strategy:
            continue
        for func_key, by_group in by_func.items():
            func_id, dim = func_key[0], func_key[1]
            if dim != dimension:
                continue
            n_total = 0
            nan_feats = []
            for group_name, feats in by_group.items():
                for fname, v in feats.items():
                    n_total += 1
                    if _is_nan(v):
                        nan_feats.append(fname)
                        feat_counter[(st, size)][fname] += 1
            records.append({
                "strategy": st,
                "size": size,
                "func": func_id,
                "n_nan": len(nan_feats),
                "n_cells": n_total,
                "frac_nan": (len(nan_feats) / n_total) if n_total else np.nan,
                "nan_features": ", ".join(sorted(nan_feats)),
            })

    detail = pd.DataFrame.from_records(records)
    if detail.empty:
        cols = ["strategy", "size", "func", "n_nan", "n_cells", "frac_nan", "nan_features"]
        return pd.DataFrame(columns=cols), pd.DataFrame()

    detail = detail.sort_values(["strategy", "size", "func"]).reset_index(drop=True)

    # ---- summary at (strategy, size) ----
    sum_rows = []
    for (st, size), grp in detail.groupby(["strategy", "size"], sort=True):
        n_nan = int(grp["n_nan"].sum())
        n_cells = int(grp["n_cells"].sum())
        affected = feat_counter[(st, size)]
        top = affected.most_common(top_n_features)
        sum_rows.append({
            "strategy": st,
            "size": size,
            "n_nan": n_nan,
            "n_cells": n_cells,
            "frac_nan": (n_nan / n_cells) if n_cells else np.nan,
            "n_funcs_affected": int((grp["n_nan"] > 0).sum()),
            "n_funcs_total": int(len(grp)),
            "n_features_affected": int(len(affected)),
            "top_features": ", ".join(f"{f} ({c})" for f, c in top) if top else "",
        })
    summary = (pd.DataFrame.from_records(sum_rows)
               .sort_values(["strategy", "size"]).reset_index(drop=True))

    if verbose:
        tot = int(detail["n_nan"].sum())
        cells = int(detail["n_cells"].sum())
        n_configs_affected = int((summary["n_nan"] > 0).sum())
        n_rows_affected = int((detail["n_nan"] > 0).sum())
        print(f"dim {dimension}: {tot} NaN ICC cells / {cells} ({tot / cells:.2%}); "
              f"{n_configs_affected}/{len(summary)} (strategy, size) configs and "
              f"{n_rows_affected} (strategy, size, func) rows affected.")

    if only_nan:
        detail = detail[detail["n_nan"] > 0].reset_index(drop=True)

    return detail, summary