import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score, confusion_matrix
import seaborn as sns

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def _get_preds_domains(trainer, loader):
    """Helper function to extract predicted labels for each domain individually"""
    _, logits, labels = trainer.evaluate(loader)

    all_domains = []

    for _, _, domains in loader:
        all_domains.extend(domains)
    all_domains = np.array(all_domains)

    preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
    labels_flat = labels.long().squeeze(1)

    return preds, labels_flat, all_domains


def compute_domain_accuracies(trainer, train_loader, val_loader):
    domain_accuracies = {}

    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        preds, labels_flat, all_domains = _get_preds_domains(trainer, loader)

        for domain in np.unique(all_domains):
            idx = np.where(all_domains == domain)[0]
            acc = (preds[idx] == labels_flat[idx]).float().mean().item()
            domain_accuracies.setdefault(domain, {})[split_name] = acc

    return domain_accuracies


def compute_domain_balanced_accuracies(trainer, loader):
    domain_bal_accuracies = {}

    preds, labels_flat, all_domains = _get_preds_domains(trainer, loader)

    for domain in np.unique(all_domains):
        idx = np.where(all_domains == domain)[0]
        bal_acc = balanced_accuracy_score(labels_flat[idx].numpy(), preds[idx].numpy())
        domain_bal_accuracies[domain] = bal_acc

    return domain_bal_accuracies


def save_accuracy_table(domain_accuracies, filename="tables/accuracy_table.csv"):
    rows = []
    for domain, acc in domain_accuracies.items():
        rows.append({"domain": domain, "train_acc": acc["train"], "val_acc": acc["val"]})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved accuracy table -> {path}")
    return df


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

def compute_classification_reports(trainer, splits):
    """
    splits: list of (split_name, loader) tuples
    
    """

    results = {}

    for split_name, loader in splits:
        preds, labels_flat, all_domains = _get_preds_domains(trainer, loader)

        overall_report = classification_report(labels_flat.numpy(), preds.numpy(), output_dict=True)
        overall_report["accuracy"] = accuracy_score(labels_flat.numpy(), preds.numpy())
        overall_report["balanced_accuracy"] = balanced_accuracy_score(labels_flat.numpy(), preds.numpy())

        domain_reports = {}
        for domain in np.unique(all_domains):
            idx = np.where(all_domains == domain)[0]
            report = classification_report(labels_flat[idx].numpy(), preds[idx].numpy(), output_dict=True)
            report["accuracy"] = accuracy_score(labels_flat[idx].numpy(), preds[idx].numpy())
            report["balanced_accuracy"] = balanced_accuracy_score(labels_flat[idx].numpy(), preds[idx].numpy())

            domain_reports[domain] = report

        results[split_name] = {"overall": overall_report, "domains": domain_reports}
    return results

def save_confusion_matrices(trainer, splits, output_dir = "plots/confusion_matrices"):
    filename = os.path.join(OUT_DIR, output_dir)
    os.makedirs(filename, exist_ok=True)

    for split_name, loader in splits:
        preds, labels_flat, all_domains = _get_preds_domains(trainer, loader)

        cm = confusion_matrix(labels_flat.numpy(), preds.numpy(), normalize="true")
        sns.heatmap(cm, fmt=".2f", annot=True, cmap="Blues")
        plt.title(f"{split_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(filename, f"{split_name}_cf_matrix.png"))
        plt.show()
        plt.close()

        for domain in np.unique(all_domains):
            idx = np.where(all_domains == domain)[0]
            cm = confusion_matrix(labels_flat[idx].numpy(), preds[idx].numpy(), normalize="true")
            sns.heatmap(cm, fmt=".2f", annot=True, cmap="Blues")
            plt.title(f"{split_name} {domain}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(os.path.join(filename, f"{split_name}_{domain}_cf_matrix.png"))
            plt.show()
            plt.close()
