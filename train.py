import torch
import json
import numpy as np
import random
import pickle

from src.datasets_classes import MultiDomainDataset
from src.utils.evaluation import (compute_domain_accuracies, compute_domain_balanced_accuracies, compute_classification_reports,
                                  save_accuracy_table, save_loss_curve, save_accuracy_curve, save_confusion_matrices,
                                  save_combined_confusion_matrix, plot_domain_metric_bar)
from src.utils.train_utils import run_training
from src.config import MODEL_OUTPUT_DIM, MODEL_OUTPUT_DIM_FRAME, NATIVE_MODEL_BY_DOMAIN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

RUN = "_w2v"

if RUN == "_w2v":

    train_dataset = MultiDomainDataset("outputs/embeddings/utterance/train", native_only=True, w2v=True)
    val_dataset = MultiDomainDataset("outputs/embeddings/utterance/validation", native_only=True, w2v=True)

    input_dims = MODEL_OUTPUT_DIM_FRAME
    native_model = NATIVE_MODEL_BY_DOMAIN
    native_model["soundscapes"] = "wav2vec"

else:
    train_dataset = MultiDomainDataset("outputs/embeddings/utterance/train", native_only=True, w2v=False)
    val_dataset = MultiDomainDataset("outputs/embeddings/utterance/validation", native_only=True, w2v=False)

    input_dims = MODEL_OUTPUT_DIM
    native_model = NATIVE_MODEL_BY_DOMAIN


domain_dims = {domain: input_dims[model] for domain, model in native_model.items()}
shared_dim = 64


with open(f"outputs/optuna/best_params{RUN}.json") as f:
    best_params = json.load(f)

trainer, train_loader, val_loader, history, fitted_pcas = run_training(train_dataset=train_dataset, val_dataset=val_dataset, input_dims=domain_dims,
                                                          shared_dim=shared_dim, max_epochs = 100, patience = 25, SEED = SEED, **best_params)

with open(f"outputs/models/fitted_pcas{RUN}.pkl", "wb") as f:
    pickle.dump(fitted_pcas, f)

with open(f"outputs/models/history{RUN}.json", "w") as f:
    json.dump(history, f)

torch.save({
    "projectors_state_dict": trainer.projectors.state_dict(),
    "classifier_state_dict": trainer.classifier.state_dict(),
    "input_dims": input_dims,
    "shared_dim": shared_dim,
    "hidden_dim": best_params["hidden_dim"],
    "dropout": best_params["dropout"],
}, f"outputs/models/best_model{RUN}.pt")


save_loss_curve(history, title="Train vs Val Loss", filename=f"plots/loss_curve{RUN}.png")

domains_accuracies = compute_domain_accuracies(trainer, train_loader, val_loader)
save_accuracy_table(domains_accuracies)

domains_bal_accs = compute_domain_balanced_accuracies(trainer, val_loader)
print(domains_bal_accs)

save_accuracy_curve(history["train_acc"], history["val_acc"], title="Train vs Val Accuracy", filename=f"plots/accuracy_curve{RUN}.png")
save_accuracy_curve(history["train_bal_acc"], history["val_bal_acc"], title="Train vs Val Balanced Accuracy", filename=f"plots/bal_acc_curve{RUN}.png")

splits = [("train", train_loader), ("val", val_loader)]

reports = compute_classification_reports(trainer, splits)
with open(f"outputs/tables/classification_reports{RUN}.json", "w") as f:
    json.dump(reports, f, indent=2)

#save_confusion_matrices(trainer, splits)
#save_combined_confusion_matrix(trainer=trainer, loader=train_loader, split_name="train") #saving combined (domains) conf. mat. for training set
#save_combined_confusion_matrix(trainer=trainer, loader=val_loader, split_name="val") #saving combined (domains) conf. mat. for validation set

plot_domain_metric_bar(reports=reports, split_name="train", filename= f"plots/train_domain_summary{RUN}.png")
plot_domain_metric_bar(reports=reports, split_name="val", filename= f"plots/val_domain_summary{RUN}.png")



