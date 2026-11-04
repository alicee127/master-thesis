import argparse
import torch
import json
import numpy as np
import random
import os

from src.utils.train_utils import run_training_no_pca
from src.utils.evaluation import save_loss_curve, save_accuracy_curve
from src.datasets_classes import MultiDomainDataset
from src.utils.results import save_split_results
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


train_dataset = MultiDomainDataset("outputs/embeddings/utterance/train", model_filter=model_name)
val_dataset = MultiDomainDataset("outputs/embeddings/utterance/validation", model_filter=model_name)

shared_dim = MODEL_OUTPUT_DIM[model_name]

with open(f"outputs/optuna/{model_name}/best_params_{model_name}.json") as f:
    best_params = json.load(f)

trainer, train_loader, train_eval_loader, val_loader, history = run_training_no_pca(train_dataset=train_dataset, val_dataset=val_dataset,
                                                                 shared_dim=shared_dim, max_epochs = 100, patience = 25,
                                                                 SEED = SEED, **best_params)


results_dir = f"outputs/results/utterance_single/{model_name}"
plots_output_dir = f"outputs/plots/utterance_single/{model_name}"
model_output_dir = f"outputs/models/utterance_single/{model_name}"
for d in (plots_output_dir, model_output_dir, results_dir):
    os.makedirs(d, exist_ok=True)

with open(f"outputs/results/utterance_single/{model_name}/history.json", "w") as f:
    json.dump(history, f)

torch.save({
    "projectors_state_dict": trainer.projectors.state_dict(),
    "classifier_state_dict": trainer.classifier.state_dict(),
    "shared_dim": shared_dim,
    "hidden_dim": best_params["hidden_dim"],
    "dropout": best_params["dropout"],
}, f"outputs/models/utterance_single/{model_name}/best_model.pt")

#save plots, results and model config
save_loss_curve(history, title=f"{model_name} Train vs Val Loss", filename=f"plots/utterance_single/{model_name}/loss_curve.png")

save_accuracy_curve(history["train_acc"], history["val_acc"], title=f"{model_name} Train vs Val Accuracy", filename=f"plots/utterance_single/{model_name}/accuracy_curve.png")
save_accuracy_curve(history["train_bal_acc"], history["val_bal_acc"], title=f"{model_name} Train vs Val Balanced Accuracy", filename=f"plots/utterance_single/{model_name}/bal_acc_curve.png")

save_split_results(trainer, train_eval_loader, "train", results_dir)
save_split_results(trainer, val_loader, "validation", results_dir)

with open(os.path.join(results_dir, "config.json"), "w") as f:
    json.dump({"model": model_name, "seed": SEED, "shared_dim": shared_dim,
               "max_epochs": 100, "patience": 25, **best_params}, f, indent=2)