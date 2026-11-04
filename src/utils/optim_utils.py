import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

def _get_preds_domains(trainer, loader):
    """
    Helper function to extract predicted labels for each domain individually

    trainer: Trainer object or None if frame-level experiment
    loader: DataLoader object for utterance-level, path of .pt files with results for frame-level
    """
    if trainer is None: #frame-level
        data = torch.load(loader, map_location="cpu")
        logits = data["logits"]
        labels = data["labels"]
        all_domains = np.array(data["domains"])

    else: #utterance-level
        _, logits, labels = trainer.evaluate(loader)

        all_domains = []

        for _, _, domains in loader:
            all_domains.extend(domains)
        all_domains = np.array(all_domains)

    preds = (torch.sigmoid(logits) > 0.5).long().reshape(-1)
    labels_flat = labels.long().reshape(-1)

    return preds, labels_flat, all_domains


# balanced accuracy computation for optimization
def compute_domain_balanced_accuracies(trainer, loader):
    domain_bal_accuracies = {}

    preds, labels_flat, all_domains = _get_preds_domains(trainer, loader)

    for domain in np.unique(all_domains):
        idx = np.where(all_domains == domain)[0]
        bal_acc = balanced_accuracy_score(labels_flat[idx].numpy(), preds[idx].numpy())
        domain_bal_accuracies[domain] = bal_acc

    return domain_bal_accuracies