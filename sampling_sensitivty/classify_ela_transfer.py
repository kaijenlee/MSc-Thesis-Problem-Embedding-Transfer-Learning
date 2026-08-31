"""
Cross-budget transfer classification for ELA features.

Trains on one sampling config and tests on ANOTHER config of the same strategy
and dimension -- by default the richest budget (size 100).

WHY
---
In the within-config experiment, train and test features come from the same
budget, so a low accuracy at size 25 could mean either

    (a) the model was trained on noisy features, or
    (b) the model was evaluated on noisy features.

Fixing the test set at one budget removes (b), so the x-axis becomes purely
"how much TRAINING budget do I need".

    --test-size 100   train cheap -> test rich
                      "how much training budget do I need, given good features
                       at deployment?"
    --train-size 100  train rich -> test cheap   (--direction deploy)
                      "if I can train once with a large budget, how cheaply can
                       I deploy?"  -- usually the more practical question for
                      algorithm selection.

CAUTION: ELA features are BIASED at small budgets, not merely noisy. A feature
computed from 50 points has a different EXPECTED VALUE than the same feature
from 500 points, so this is a genuine distribution shift, not just a change in
variance. An accuracy drop therefore mixes "trained on noisy data" with
"train/test mismatch". Running both directions helps separate them: a symmetric
drop points to mismatch, an asymmetric one to training-data quality.

The train/test split is at INSTANCE level and is identical across configs (it
depends only on the labels), so instance i is on the same side of the split in
both the train and the test config -- no instance is ever seen in both.

Usage:
  python classify_ela_transfer.py /path/to/ela/dir --dimension 5
  python classify_ela_transfer.py /path/to/ela/dir --test-size 100
  python classify_ela_transfer.py /path/to/ela/dir --direction deploy
  python classify_ela_transfer.py /path/to/ela/dir --strategies sobol ilhs
"""

import argparse
import gc
import os
import pickle
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration (kept identical to the within-config experiment)
# ---------------------------------------------------------------------------

N_FUNCTIONS = 24
N_INSTANCES = 100
N_RUNS = 30
N_FOLDS = 5
N_TRAIN_PER_FOLD = 20
RANDOM_STATE = 42

DEFAULT_N_INSTANCES_TRAIN = [1, 2, 3, 5, 7, 10, 15, 20]
DEFAULT_N_RUNS_TRAIN = [1, 2, 3, 5]

RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_MIN_SAMPLES_LEAF = 1

OMIT_GROUPS = {"levelset"}

STRATEGIES = ["cma_random", "ilhs", "lhs", "lhs_random_cd", "sobol", "uniform"]
SIZES = [25, 50, 75, 100]

ELA_FEATURE_GROUPS = {
    "ela_dist": [
        "ela_distr.skewness", "ela_distr.kurtosis",
        "ela_distr.number_of_peaks", "ela_distr.costs_runtime",
    ],
    "meta": [
        "ela_meta.lin_simple.adj_r2", "ela_meta.lin_simple.intercept",
        "ela_meta.lin_simple.coef.min", "ela_meta.lin_simple.coef.max",
        "ela_meta.lin_simple.coef.max_by_min", "ela_meta.lin_w_interact.adj_r2",
        "ela_meta.quad_simple.adj_r2", "ela_meta.quad_simple.cond",
        "ela_meta.quad_w_interact.adj_r2", "ela_meta.costs_runtime",
    ],
    "disp": [
        "disp.ratio_mean_02", "disp.ratio_mean_05", "disp.ratio_mean_10",
        "disp.ratio_mean_25", "disp.ratio_median_02", "disp.ratio_median_05",
        "disp.ratio_median_10", "disp.ratio_median_25", "disp.diff_mean_02",
        "disp.diff_mean_05", "disp.diff_mean_10", "disp.diff_mean_25",
        "disp.diff_median_02", "disp.diff_median_05", "disp.diff_median_10",
        "disp.diff_median_25", "disp.costs_runtime",
    ],
    "nbc": [
        "nbc.nn_nb.sd_ratio", "nbc.nn_nb.mean_ratio", "nbc.nn_nb.cor",
        "nbc.dist_ratio.coeff_var", "nbc.nb_fitness.cor", "nbc.costs_runtime",
    ],
    "ic": [
        "ic.h_max", "ic.eps_s", "ic.eps_max", "ic.eps_ratio",
        "ic.m0", "ic.costs_runtime",
    ],
    "pca": [
        "pca.expl_var.cov_x", "pca.expl_var.cor_x", "pca.expl_var.cov_init",
        "pca.expl_var.cor_init", "pca.expl_var_PC1.cov_x",
        "pca.expl_var_PC1.cor_x", "pca.expl_var_PC1.cov_init",
        "pca.expl_var_PC1.cor_init", "pca.costs_runtime",
    ],
}

CANDIDATE_FEATURES = [(g, f) for g, feats in ELA_FEATURE_GROUPS.items()
                      if g not in OMIT_GROUPS
                      for f in feats if "costs_runtime" not in f]
N_CANDIDATE_FEATURES = len(CANDIDATE_FEATURES)


def get_omit_features(dimension):
    if dimension == 2:
        return {'disp.diff_median_02', 'disp.ratio_median_02',
                'disp.ratio_mean_02', 'ela_meta.quad_simple.cond',
                'disp.diff_mean_02', 'ic.eps_ratio'}
    elif dimension in (5, 10):
        return {'ela_meta.quad_simple.cond'}
    return set()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_arrays(data, dimension):
    """(median, runs, labels) over ALL candidate features -- no dropping here.

    Feature selection is deferred so that the train and test configs can be
    reduced to a COMMON feature set; dropping independently would leave the two
    matrices with different columns.
    """
    n_total = N_FUNCTIONS * N_INSTANCES
    med = np.empty((n_total, N_CANDIDATE_FEATURES))
    runs = np.empty((n_total, N_RUNS, N_CANDIDATE_FEATURES))
    labels = np.empty(n_total, dtype=int)

    for fi in range(N_FUNCTIONS):
        for ii in range(N_INSTANCES):
            row = fi * N_INSTANCES + ii
            key = (fi + 1, ii + 1, dimension)
            if key not in data:
                raise KeyError(f"missing {key} in ELA pickle")
            inst = data[key]
            labels[row] = fi
            for k, (grp, feat) in enumerate(CANDIDATE_FEATURES):
                vals = [inst[grp][r][feat] for r in range(N_RUNS)]
                med[row, k] = np.nanmedian(vals)
                runs[row, :, k] = vals
    return med, runs, labels


def common_feature_mask(runs_a, runs_b, omit_features, drop_nan=True):
    """Columns kept for BOTH configs: finite everywhere in each, not omitted."""
    keep = np.ones(N_CANDIDATE_FEATURES, dtype=bool)
    for k, (_, feat) in enumerate(CANDIDATE_FEATURES):
        if feat in omit_features:
            keep[k] = False
    if drop_nan:
        for arr in (runs_a, runs_b):
            flat = arr.reshape(-1, N_CANDIDATE_FEATURES)
            keep &= np.all(np.isfinite(flat), axis=0)
    return keep


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def run_transfer(train_runs, test_med, test_runs, labels,
                 n_instances_train, n_runs_train):
    """5-fold CV: training rows from `train_runs`, test rows from `test_*`.

    The fold split depends only on `labels`, so it is IDENTICAL for both
    configs -- instance i sits on the same side of the split in each, and no
    instance is ever both trained on and tested on.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    acc_median, acc_runs, consistency = [], [], []
    all_true, all_pred_med, all_pred_runs, all_cons = [], [], [], []
    sampled_instances, sampled_runs_all = [], []

    for fold_idx, (test_idx, train_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels)):
        # NESTED subsampling, so the sweep is PAIRED and monotone-comparable.
        # A fixed permutation per (fold, class) is truncated to the requested
        # size, which makes n_instances_train=5 a strict SUBSET of =10. Drawing
        # independently per grid point would move the curve through resampling
        # noise as well as through the effect being measured. The same holds for
        # runs: the permutation is seeded per (fold, instance), so it does not
        # depend on how many instances or runs the grid point asks for.
        train_labels = labels[train_idx]
        chosen = []
        for cls in range(N_FUNCTIONS):
            cls_idx = train_idx[train_labels == cls]
            perm = np.random.default_rng(
                (RANDOM_STATE, fold_idx, cls)).permutation(cls_idx)
            chosen.append(perm[:min(n_instances_train, len(perm))])
        sel = np.concatenate(chosen)
        n_train = len(sel)
        sampled_instances.append(sel.copy())

        k = min(n_runs_train, N_RUNS)
        sruns = np.stack([
            np.random.default_rng(
                (RANDOM_STATE, fold_idx, int(row))).permutation(N_RUNS)[:k]
            for row in sel])
        sampled_runs_all.append(sruns)

        F = train_runs.shape[2]
        X_train = train_runs[sel][np.arange(n_train)[:, None], sruns, :]
        X_train = X_train.reshape(n_train * k, F)
        y_train = np.repeat(labels[sel], k)

        scaler = StandardScaler()
        Xtr = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)

        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
            max_features=RF_MAX_FEATURES, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(Xtr, y_train)
        del X_train, Xtr

        y_test = labels[test_idx]

        # --- test on the median feature vector (denoised) ---
        Xte = np.nan_to_num(scaler.transform(test_med[test_idx]), nan=0.0)
        pred_med = rf.predict(Xte)
        acc_median.append(accuracy_score(y_test, pred_med))
        all_pred_med.extend(pred_med)

        # --- test on individual runs (batched: one predict call) ---
        nt = len(test_idx)
        Xr = test_runs[test_idx].reshape(nt * N_RUNS, F)
        Xr = np.nan_to_num(scaler.transform(Xr), nan=0.0)
        pred_r = rf.predict(Xr).reshape(nt, N_RUNS)

        cons = (pred_r == y_test[:, None]).mean(axis=1)
        acc_runs.append(float((pred_r == y_test[:, None]).mean()))
        consistency.append(float(cons.mean()))
        all_cons.extend(cons)
        all_pred_runs.extend(pred_r)
        all_true.extend(y_test)

    all_true = np.array(all_true)
    all_pred_runs = np.array(all_pred_runs)
    mv = mode(all_pred_runs, axis=1, keepdims=False).mode

    return {
        "fold_accuracies_median": np.array(acc_median),
        "fold_accuracies_all_runs": np.array(acc_runs),
        "fold_consistency": np.array(consistency),
        "per_instance_consistency": np.array(all_cons),
        "true_labels": all_true,
        "pred_median": np.array(all_pred_med),
        "pred_runs": all_pred_runs,
        "pred_majority_vote": mv,
        "overall_accuracy_median": accuracy_score(all_true,
                                                  np.array(all_pred_med)),
        "overall_accuracy_all_runs": float(np.mean(acc_runs)),
        "overall_accuracy_majority_vote": accuracy_score(all_true, mv),
        "overall_consistency_mean": float(np.mean(all_cons)),
        "overall_consistency_std": float(np.std(all_cons)),
        "sampled_instances": sampled_instances,
        "sampled_runs": sampled_runs_all,
        "n_instances_train": n_instances_train,
        "n_runs_train": n_runs_train,
    }


def _worker(train_runs, test_med, test_runs, labels, meta, ni, nr):
    res = run_transfer(train_runs, test_med, test_runs, labels, ni, nr)
    res.update(meta)
    res["n_feval_train"] = (N_FUNCTIONS * ni * nr
                            * meta["train_size"] * meta["dimension"])
    return res


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

_SENTINEL = None


def _writer_loop(q, path):
    with h5py.File(path, "a") as out:
        while True:
            res = q.get()
            if res is _SENTINEL:
                q.task_done()
                break
            grp_name = f"{res['strategy']}_train{res['train_size']}_test{res['test_size']}"
            g = out.require_group(grp_name)
            sub = f"inst_{res['n_instances_train']:02d}_runs_{res['n_runs_train']:02d}"
            if sub in g:
                q.task_done()
                continue
            s = g.create_group(sub)
            for k in ("fold_accuracies_median", "fold_accuracies_all_runs",
                      "fold_consistency", "per_instance_consistency",
                      "true_labels", "pred_median", "pred_runs",
                      "pred_majority_vote"):
                s.create_dataset(k, data=res[k])
            for name, lst in (("sampled_instances", res["sampled_instances"]),
                              ("sampled_runs", res["sampled_runs"])):
                sg = s.create_group(name)
                for i, arr in enumerate(lst):
                    sg.create_dataset(f"fold_{i}", data=arr)
            for k in ("n_instances_train", "n_runs_train", "n_feval_train",
                      "strategy", "train_size", "test_size", "dimension",
                      "n_features", "overall_accuracy_median",
                      "overall_accuracy_all_runs",
                      "overall_accuracy_majority_vote",
                      "overall_consistency_mean", "overall_consistency_std"):
                s.attrs[k] = res[k]
            out.flush()
            print(f"  [written] {grp_name}/{sub}  "
                  f"median={res['overall_accuracy_median']:.4f}  "
                  f"runs={res['overall_accuracy_all_runs']:.4f}  "
                  f"mv={res['overall_accuracy_majority_vote']:.4f}  "
                  f"fevals={res['n_feval_train']}")
            q.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_dir, output_dir=None, dimension=5, strategies=None,
         train_sizes=None, test_size=100, direction="train",
         n_instances_train_list=None, n_runs_train_list=None,
         max_workers=None, drop_nan_features=True):
    """direction="train"  : sweep the TRAIN budget, hold the test budget fixed
       direction="deploy" : hold the TRAIN budget fixed, sweep the TEST budget
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir or input_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_file = output_dir / f"ela_classification_transfer_dim{dimension}.h5"

    strategies = strategies or STRATEGIES
    train_sizes = train_sizes or SIZES
    nis = n_instances_train_list or DEFAULT_N_INSTANCES_TRAIN
    nrs = n_runs_train_list or DEFAULT_N_RUNS_TRAIN
    grid = [(a, b) for a in nis for b in nrs]
    omit = get_omit_features(dimension)
    max_workers = max_workers or min(len(grid), os.cpu_count() or 4)

    # build the (train_size, test_size) pairs
    if direction == "train":
        pairs = [(s, test_size) for s in train_sizes]
        swept = "TRAIN budget swept, test budget fixed"
    elif direction == "deploy":
        pairs = [(test_size, s) for s in train_sizes]
        swept = "TRAIN budget fixed, TEST budget swept"
    else:
        raise ValueError("direction must be 'train' or 'deploy'")

    print(f"Cross-budget transfer classification — dim {dimension}")
    print(f"  {swept}")
    print(f"  strategies: {strategies}")
    print(f"  (train, test) size pairs: {pairs}")
    print(f"  grid points per pair: {len(grid)}   workers: {max_workers}")
    if omit:
        print(f"  dimension omissions: {sorted(omit)}")
    print()

    q = queue.Queue()
    t = threading.Thread(target=_writer_loop, args=(q, str(out_file)),
                         daemon=True)
    t.start()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for strategy in strategies:
            for tr_size, te_size in pairs:
                f_tr = input_dir / f"{strategy}_{tr_size}_ela.pkl"
                f_te = input_dir / f"{strategy}_{te_size}_ela.pkl"
                if not f_tr.exists() or not f_te.exists():
                    print(f"WARNING: missing pickle for {strategy} "
                          f"{tr_size}/{te_size}, skipping.")
                    continue

                print(f"\n{'='*64}\n{strategy}: train {tr_size} -> test {te_size}"
                      f"\n{'='*64}")
                with open(f_tr, "rb") as fh:
                    _, tr_runs, labels = build_arrays(pickle.load(fh), dimension)
                if te_size == tr_size:
                    te_med = np.nanmedian(tr_runs, axis=1)
                    te_runs = tr_runs
                else:
                    with open(f_te, "rb") as fh:
                        te_med, te_runs, lab2 = build_arrays(pickle.load(fh),
                                                             dimension)
                    assert np.array_equal(labels, lab2), "label mismatch"

                keep = common_feature_mask(tr_runs, te_runs, omit,
                                           drop_nan=drop_nan_features)
                tr_runs = tr_runs[:, :, keep]
                te_runs = te_runs[:, :, keep]
                te_med = te_med[:, keep]
                dropped = [f for (g, f), k in zip(CANDIDATE_FEATURES, keep) if not k]
                print(f"  features kept: {int(keep.sum())}/{N_CANDIDATE_FEATURES}"
                      + (f"   dropped: {dropped}" if dropped else ""))

                meta = dict(strategy=strategy, train_size=tr_size,
                            test_size=te_size, dimension=dimension,
                            n_features=int(keep.sum()))
                futs = {pool.submit(_worker, tr_runs, te_med, te_runs, labels,
                                    meta, ni, nr): (ni, nr)
                        for ni, nr in grid}
                for fut in as_completed(futs):
                    ni, nr = futs[fut]
                    try:
                        q.put(fut.result())
                    except Exception as exc:
                        print(f"  ERROR inst_{ni:02d}_runs_{nr:02d}: {exc}")
                del tr_runs, te_runs, te_med
                gc.collect()

    q.put(_SENTINEL)
    q.join()
    t.join()
    print(f"\nResults saved to: {out_file}")
    print("Structure: {strategy}_train{S}_test{S}/inst_XX_runs_YY/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train on one sampling budget, test on another.")
    p.add_argument("input_dir", type=str)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--dimension", type=int, default=5)
    p.add_argument("--strategies", nargs="+", default=None)
    p.add_argument("--train-sizes", nargs="+", type=int, default=None,
                   help="budgets to sweep (default 25 50 75 100)")
    p.add_argument("--test-size", type=int, default=100,
                   help="the FIXED budget (test budget when direction=train, "
                        "train budget when direction=deploy). Default 100.")
    p.add_argument("--direction", choices=["train", "deploy"], default="train",
                   help="'train': sweep training budget, fixed rich test set. "
                        "'deploy': train once at --test-size, sweep the test "
                        "budget.")
    p.add_argument("--n-instances-train", nargs="+", type=int, default=None)
    p.add_argument("--n-runs-train", nargs="+", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--no-drop-nan-features", action="store_true")
    a = p.parse_args()
    main(a.input_dir, a.output_dir, a.dimension, a.strategies, a.train_sizes,
         a.test_size, a.direction, a.n_instances_train, a.n_runs_train,
         a.max_workers, not a.no_drop_nan_features)