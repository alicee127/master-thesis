import torch
import json
import numpy as np
import random
import pickle

from src.datasets_classes import MultiDomainDataset
from src.utils.evaluation import (compute_domain_accuracies, compute_domain_balanced_accuracies, compute_classification_reports,
                                  save_accuracy_table, save_loss_curve, save_accuracy_curve, save_confusion_matrices)
from src.utils.train_utils import run_training

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


train_dataset = MultiDomainDataset("outputs/embeddings/train")
val_dataset = MultiDomainDataset("outputs/embeddings/validation")

input_dims = {"speech": 1024, "music": 1024, "animal": 768, "soundscapes": 512}
shared_dim = 64


with open("outputs/optuna/best_params.json") as f:
    best_params = json.load(f)

trainer, train_loader, val_loader, history, fitted_pcas = run_training(train_dataset=train_dataset, val_dataset=val_dataset, input_dims=input_dims,
                                                          shared_dim=shared_dim, max_epochs = 100, patience = 100, SEED = SEED, **best_params)

with open("outputs/models/fitted_pcas.pkl", "wb") as f:
    pickle.dump(fitted_pcas, f)

with open("outputs/models/history.json", "w") as f:
    json.dump(history, f)

torch.save({
    "projectors_state_dict": trainer.projectors.state_dict(),
    "classifier_state_dict": trainer.classifier.state_dict(),
    "input_dims": input_dims,
    "shared_dim": shared_dim,
    "hidden_dim": best_params["hidden_dim"],
    "dropout": best_params["dropout"],
}, "outputs/models/best_model.pt")


save_loss_curve(history, title="Train vs Val Loss", filename="plots/loss_curve.png")

domains_accuracies = compute_domain_accuracies(trainer, train_loader, val_loader)
save_accuracy_table(domains_accuracies)

domains_bal_accs = compute_domain_balanced_accuracies(trainer, val_loader)
print(domains_bal_accs)

save_accuracy_curve(history["train_acc"], history["val_acc"], title="Train vs Val Accuracy", filename="plots/accuracy_curve.png")
save_accuracy_curve(history["train_bal_acc"], history["val_bal_acc"], title="Train vs Val Balanced Accuracy", filename="plots/bal_acc_curve.png")

splits = [("train", train_loader), ("val", val_loader)]

reports = compute_classification_reports(trainer, splits)
with open("outputs/tables/classification_reports.json", "w") as f:
    json.dump(reports, f, indent=2)

save_confusion_matrices(trainer, splits)

