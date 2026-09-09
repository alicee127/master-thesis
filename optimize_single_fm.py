import argparse
import optuna
import torch
import numpy as np
import random
import json
import os

from src.datasets_classes import MultiDomainDataset
from src.utils.train_utils import run_training_no_pca
from src.utils.evaluation import compute_domain_balanced_accuracies
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


def objective(trial):
    #define search space

    #shared_dim = trial.suggest_categorical("shared_dim", [64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    trainer, train_loader, val_loader, history = run_training_no_pca(train_dataset, val_dataset,
                                                shared_dim=shared_dim, hidden_dim=hidden_dim, lr=lr, dropout=dropout,
                                                max_epochs = 100, patience = 10, SEED = SEED)

    domain_bal_accs = compute_domain_balanced_accuracies(trainer, val_loader)
    print(f"Trial {trial.number} per-domain balanced accuracy: {domain_bal_accs}")
    trial.set_user_attr("domain_bal_accs", domain_bal_accs)

    mean_bal_acc = sum(domain_bal_accs.values()) / len(domain_bal_accs)


    return mean_bal_acc

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best params:", study.best_params)
    print("Best mean balanced accuracy:", study.best_value)

    out_dir = f"outputs/optuna/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"best_params_{model_name}.json"), "w") as f:
        json.dump(study.best_params, f)

    study.trials_dataframe().to_csv(os.path.join(out_dir, f"trials_{model_name}.csv"), index = False)