import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_results(h5_path):
    data = {}
    with h5py.File(h5_path, "r") as f:
        for config in f.keys():
            grp = f[config]
            data[config] = {
                "fold_mean": grp["fold_accuracies_mean"][:],
                "fold_runs": grp["fold_accuracies_all_runs"][:],
                "fold_consistency": grp["fold_consistency"][:],
                "per_instance_consistency": grp["per_instance_consistency"][:],
                "overall_mean": grp.attrs["overall_accuracy_mean"],
                "overall_runs": grp.attrs["overall_accuracy_all_runs"],
                "consistency_mean": grp.attrs["overall_consistency_mean"],
                "consistency_std": grp.attrs["overall_consistency_std"],
            }
    return data

def plot_overall_accuracies(results):
    configs = list(results.keys())
    mean_acc = [results[c]["overall_mean"] for c in configs]
    run_acc = [results[c]["overall_runs"] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, mean_acc, width, label="Mean features")
    plt.bar(x + width/2, run_acc, width, label="All runs")

    plt.xticks(x, configs, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Overall Accuracy per Configuration")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fold_distributions(results):
    configs = list(results.keys())

    data_mean = [results[c]["fold_mean"] for c in configs]
    data_runs = [results[c]["fold_runs"] for c in configs]

    plt.figure(figsize=(14, 6))

    plt.boxplot(data_mean, positions=np.arange(len(configs)) - 0.2, widths=0.3)
    plt.boxplot(data_runs, positions=np.arange(len(configs)) + 0.2, widths=0.3)

    plt.xticks(np.arange(len(configs)), configs, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Fold Accuracy Distribution (Mean vs All Runs)")
    plt.legend(["Mean features", "All runs"])
    plt.tight_layout()
    plt.show()

def plot_consistency_hist(results, config_key, bins=20):
    consistency = results[config_key]["per_instance_consistency"]

    plt.figure(figsize=(7, 5))
    plt.hist(consistency, bins=bins)
    plt.xlabel("Per-instance consistency (fraction of correct runs)")
    plt.ylabel("Count")
    plt.title(f"Consistency Distribution: {config_key}")
    plt.tight_layout()
    plt.show()

def plot_consistency_boxplot(results):
    configs = list(results.keys())
    data = [results[c]["per_instance_consistency"] for c in configs]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data)

    plt.xticks(range(1, len(configs)+1), configs, rotation=45, ha="right")
    plt.ylabel("Consistency")
    plt.title("Per-instance Consistency Across Configurations")
    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_consistency(results):
    configs = list(results.keys())

    acc = [results[c]["overall_runs"] for c in configs]
    cons = [results[c]["consistency_mean"] for c in configs]

    plt.figure(figsize=(7, 5))
    plt.scatter(acc, cons)

    for i, c in enumerate(configs):
        plt.text(acc[i], cons[i], c, fontsize=8)

    plt.xlabel("Accuracy (all runs)")
    plt.ylabel("Mean consistency")
    plt.title("Accuracy vs Consistency")
    plt.tight_layout()
    plt.show()

def rank_configs(results, by="overall_runs"):
    sorted_items = sorted(results.items(),
                          key=lambda x: x[1][by],
                          reverse=True)

    for i, (config, vals) in enumerate(sorted_items, 1):
        print(f"{i:2d}. {config:15s} | acc={vals['overall_runs']:.4f} | cons={vals['consistency_mean']:.4f}")


def load_tla_results(h5_path):
    data = {}
    with h5py.File(h5_path, "r") as f:
        for config in f.keys():
            grp = f[config]
            data[config] = {
                "fold_mean": grp["fold_accuracies_mean"][:],
                "fold_runs": grp["fold_accuracies_all_runs"][:],
                "fold_consistency": grp["fold_consistency"][:],
                "fold_pca": grp["fold_n_pca_components"][:],
                "per_instance_consistency": grp["per_instance_consistency"][:],
                "overall_mean": grp.attrs["overall_accuracy_mean"],
                "overall_runs": grp.attrs["overall_accuracy_all_runs"],
                "consistency_mean": grp.attrs["overall_consistency_mean"],
                "consistency_std": grp.attrs["overall_consistency_std"],
            }
    return data


def plot_overall_tla_accuracies(results):
    configs = list(results.keys())
    mean_acc = [results[c]["overall_mean"] for c in configs]
    run_acc = [results[c]["overall_runs"] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - width/2, mean_acc, width, label="Mean features")
    plt.bar(x + width/2, run_acc, width, label="All runs")

    plt.xticks(x, configs, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("TLA: Overall Accuracy per Configuration")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pca_components(results):
    configs = list(results.keys())
    mean_pca = [np.mean(results[c]["fold_pca"]) for c in configs]
    std_pca = [np.std(results[c]["fold_pca"]) for c in configs]

    x = np.arange(len(configs))

    plt.figure(figsize=(12, 5))
    plt.bar(x, mean_pca, yerr=std_pca)

    plt.xticks(x, configs, rotation=45, ha="right")
    plt.ylabel("Number of PCA components")
    plt.title("TLA: PCA Components (99% variance)")
    plt.tight_layout()
    plt.show()

def rank_tla_configs(results, by="overall_mean"):
    sorted_items = sorted(results.items(),
                          key=lambda x: x[1][by],
                          reverse=True)

    for i, (config, vals) in enumerate(sorted_items, 1):
        print(f"{i:2d}. {config:15s} | acc={vals['overall_mean']:.4f} | "
              f"cons={vals['consistency_mean']:.4f}")