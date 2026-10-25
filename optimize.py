import optuna
import torch
import numpy as np
import random
import json
import os

from src.datasets_classes import MultiDomainDataset
from src.utils.train_utils import run_training
from src.utils.evaluation import compute_domain_balanced_accuracies
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


def objective(trial):
    #define search space

    #shared_dim = trial.suggest_categorical("shared_dim", [64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)


    trainer, _, val_loader, _, _ = run_training(train_dataset, val_dataset, input_dims=domain_dims,
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

    os.makedirs(f"outputs/optuna", exist_ok=True)
    with open(f"outputs/optuna/best_params{RUN}.json", "w") as f:
        json.dump(study.best_params, f)

    study.trials_dataframe().to_csv(f"outputs/optuna/trials{RUN}.csv", index = False)