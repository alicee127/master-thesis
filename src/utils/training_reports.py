import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


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


import numpy as np

def compute_domain_accuracies(trainer, train_loader, val_loader):
    domain_accuracies = {}

    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        _, logits, labels = trainer.evaluate(loader)

        all_domains = []
        for _, _, domains in loader:
            all_domains.extend(domains)
        all_domains = np.array(all_domains)

        preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1)
        labels_flat = labels.long().squeeze(1)

        for domain in np.unique(all_domains):
            idx = np.where(all_domains == domain)[0]
            acc = (preds[idx] == labels_flat[idx]).float().mean().item()
            domain_accuracies.setdefault(domain, {})[split_name] = acc

    return domain_accuracies


def save_accuracy_table(domain_accuracies, filename="tables/accuracy_table.csv"):
    rows = []
    for domain, acc in domain_accuracies.items():
        rows.append({"domain": domain, "train_acc": acc["train"], "val_acc": acc["val"]})
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved accuracy table -> {path}")
    return df
