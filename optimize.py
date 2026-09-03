import optuna
import torch
import numpy as np
import random
import json
import os

from src.datasets_classes import MultiDomainDataset
from src.utils.train_utils import run_training
from src.utils.evaluation import compute_domain_balanced_accuracies

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


def objective(trial):
    #define search space

    #shared_dim = trial.suggest_categorical("shared_dim", [64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    trainer, train_loader, val_loader, history, fitted_pcas = run_training(train_dataset, val_dataset, input_dims,
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

    os.makedirs("outputs/optuna", exist_ok=True)
    with open("outputs/optuna/best_params.json", "w") as f:
        json.dump(study.best_params, f)

    study.trials_dataframe().to_csv("outputs/optuna/trials.csv", index = False)