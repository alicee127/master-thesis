import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix

CLASS_NAMES = ["negative", "positive"]
OUT_DIR = "outputs"
CMAP = "Purples"

# GET PREDICTIONS

def _to_df(logits, labels, domains):
    prob = torch.sigmoid(logits.reshape(-1)).numpy()
    df = pd.DataFrame({
        "domain": np.asarray(domains),
        "y_true": labels.long().reshape(-1).numpy(),
        "prob": prob,
        "y_pred": (prob > 0.5).astype(int)
    })

    return df

def predictions_from_file(path):
    d = torch.load(path, map_location="cpu") #weights_only = True ??
    df = _to_df(d["logits"], d["labels"], d["domains"])
    return df


# COMPUTE METRICS

def compute_metrics(df):
    prec, rec, f1, _ = precision_recall_fscore_support(df["y_true"], df["y_pred"], average="macro", zero_division=0)
    acc = accuracy_score(df["y_true"], df["y_pred"])
    metrics_dict = {
        "n": len(df),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
    return metrics_dict

def metrics_table(df):
    rows = {dom: compute_metrics(g) for dom, g in df.groupby("domain")}
    table = pd.DataFrame.from_dict(rows, orient="index")

    metric_cols = ["accuracy", "precision", "recall", "f1"]
    mean_row = table[metric_cols].mean()
    mean_row["n"] = table["n"].sum()
    table.loc["domain average"] = mean_row
    table.loc["overall"] = pd.Series(compute_metrics(df))
    table["n"] = table["n"].astype(int)
    return table[["n"] + metric_cols]

def selection_score(df):
    #mean of bal. acc. over domains, used as a metric to choose the best model configurations on validation set
    bal_acc_mean = np.mean([balanced_accuracy_score(g["y_true"], g["y_pred"]) for _, g in df.groupby("domain")])
    return bal_acc_mean

def plot_confusion_grid(df, title = None, path = None, show = True):
    domains = sorted(df["domain"].unique())
    n_cols = 2
    n_rows = int(np.ceil(len(domains) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
    axes = axes.reshape(-1)

    for ax, dom in zip(axes, domains):
        g = df[df["domain"] == dom]

        cm = confusion_matrix(g["y_true"], g["y_pred"], labels = [0, 1], normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap = CMAP, vmin=0, vmax=1, cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{dom} (n={len(g)})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for ax in axes[len(domains):]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    plt.close(fig)


def plot_overall_confusion_grid(dfs, n_cols=5, path=None, show=True):
    names = list(dfs)
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows), squeeze=False)
    axes = axes.reshape(-1)

    for ax, name in zip(axes, names):
        df = dfs[name]
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1], normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap=CMAP, vmin=0, vmax=1, cbar=False, square=True,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for ax in axes[len(names):]:
        ax.axis("off")

    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    

def plot_domain_comparison(metrics_by_approach, metric="recall", path=None, show=True):
    #per-domain bar plot with multiple approaches
    df = pd.DataFrame({k: v[metric] for k, v in metrics_by_approach.items()})
    #df = df.drop(index="overall")

    ax = df.plot(kind="bar", figsize=(10, 5), width=0.8, colormap="tab20b")
    ax.set_ylabel("Balanced accuracy" if metric=="recall" else metric)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.legend(title=None, loc="lower right")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=6, padding=2)
    plt.xticks(rotation=0)
    plt.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    plt.close()

def save_metrics(df, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    metrics_table(df).to_csv(os.path.join(out_dir, f"{name}_metrics.csv"))



# PLOTS FOR TRAINING (loss and accuracy curves)

def save_loss_curve(history, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(len(history["train_loss"]))
    ax.plot(epochs, history["train_loss"], label="train_loss")
    ax.plot(epochs, history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved loss curve -> {path}")
    return fig


def save_accuracy_curve(variable_train, variable_val, title="Train vs Val Accuracy", filename="plots/accuracy_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(len(variable_train))
    ax.plot(epochs, variable_train, label="train_acc")
    ax.plot(epochs, variable_val, label="val_acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved accuracy curve -> {path}")
    return fig



