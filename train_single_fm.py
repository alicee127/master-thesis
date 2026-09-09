import argparse
import torch
import json
import numpy as np
import random

from src.utils.train_utils import run_training_no_pca
from src.utils.evaluation import (compute_domain_balanced_accuracies, compute_classification_reports, compute_domain_accuracies,
                                  save_loss_curve, save_accuracy_curve, save_accuracy_table, save_confusion_matrices,
                                  save_combined_confusion_matrix, plot_domain_metric_bar)
from src.datasets_classes import MultiDomainDataset
from src.config import MODEL_OUTPUT_DIM


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["e2v", "mert", "aves2", "clap"])
args = parser.parse_args()
model_name = args.model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


train_dataset = MultiDomainDataset("outputs/embeddings/train", model_filter=model_name)
val_dataset = MultiDomainDataset("outputs/embeddings/validation", model_filter=model_name)

shared_dim = MODEL_OUTPUT_DIM[model_name]

with open(f"outputs/optuna/{model_name}/best_params_{model_name}.json") as f:
    best_params = json.load(f)

trainer, train_loader, val_loader, history = run_training_no_pca(train_dataset=train_dataset, val_dataset=val_dataset,
                                                                              shared_dim=shared_dim, max_epochs = 100, patience = 25,
                                                                              SEED = SEED, **best_params)

out_prefix = f"outputs/models/{model_name}"

with open(f"{out_prefix}_history.json", "w") as f:
    json.dump(history, f)

torch.save({
    "projectors_state_dict": trainer.projectors.state_dict(),
    "classifier_state_dict": trainer.classifier.state_dict(),
    "shared_dim": shared_dim,
    "hidden_dim": best_params["hidden_dim"],
    "dropout": best_params["dropout"],
}, f"{out_prefix}_best_model.pt")


save_loss_curve(history, title=f"{model_name} Train vs Val Loss", filename=f"plots/{model_name}_loss_curve.png")

domains_accuracies = compute_domain_accuracies(trainer, train_loader, val_loader)
save_accuracy_table(domains_accuracies, f"tables/{model_name}_accuracy_table.csv")

domains_bal_accs = compute_domain_balanced_accuracies(trainer, val_loader)
print(domains_bal_accs)

save_accuracy_curve(history["train_acc"], history["val_acc"], title=f"{model_name} Train vs Val Accuracy", filename=f"plots/{model_name}_accuracy_curve.png")
save_accuracy_curve(history["train_bal_acc"], history["val_bal_acc"], title=f"{model_name} Train vs Val Balanced Accuracy", filename=f"plots/{model_name}_bal_acc_curve.png")

splits = [("train", train_loader), ("val", val_loader)]

reports = compute_classification_reports(trainer, splits)
with open(f"outputs/tables/{model_name}_classification_reports.json", "w") as f:
    json.dump(reports, f, indent=2)

save_confusion_matrices(trainer, splits, output_dir=f"plots/confusion_matrices/{model_name}")

save_combined_confusion_matrix(trainer=trainer, loader=train_loader, split_name = "train", output_dir=f"plots/confusion_matrices/{model_name}")
save_combined_confusion_matrix(trainer=trainer, loader=val_loader, split_name="val", output_dir=f"plots/confusion_matrices/{model_name}")

plot_domain_metric_bar(reports, "train", filename=f"plots/{model_name}_train_domain_summary.png")
plot_domain_metric_bar(reports, "val", filename=f"plots/{model_name}_val_domain_summary.png")


