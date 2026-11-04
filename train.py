import torch
import json
import numpy as np
import random
import pickle
import os
import argparse

from src.datasets_classes import MultiDomainDataset
from src.utils.evaluation import save_loss_curve, save_accuracy_curve
from src.utils.train_utils import run_training
from src.utils.results import save_split_results
from src.config import MODEL_OUTPUT_DIM, MODEL_OUTPUT_DIM_FRAME, NATIVE_MODEL_BY_DOMAIN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="", choices=["", "_w2v"])
args = parser.parse_args()
RUN = args.run

if RUN == "_w2v":

    train_dataset = MultiDomainDataset("outputs/embeddings/utterance/train", native_only=True, w2v=True)
    val_dataset = MultiDomainDataset("outputs/embeddings/utterance/validation", native_only=True, w2v=True)

    input_dims = MODEL_OUTPUT_DIM_FRAME
    native_model = dict(NATIVE_MODEL_BY_DOMAIN)
    native_model["soundscapes"] = "wav2vec"

else:
    train_dataset = MultiDomainDataset("outputs/embeddings/utterance/train", native_only=True, w2v=False)
    val_dataset = MultiDomainDataset("outputs/embeddings/utterance/validation", native_only=True, w2v=False)

    input_dims = MODEL_OUTPUT_DIM
    native_model = dict(NATIVE_MODEL_BY_DOMAIN)


domain_dims = {domain: input_dims[model] for domain, model in native_model.items()}
shared_dim = 64

results_dir = f"outputs/results/utterance{RUN}"
plots_output_dir = f"outputs/plots/utterance{RUN}"
model_output_dir = f"outputs/models/utterance{RUN}"
for d in (plots_output_dir, model_output_dir, results_dir):
    os.makedirs(d, exist_ok=True)


with open(f"outputs/optuna/best_params{RUN}.json") as f:
    best_params = json.load(f)

trainer, train_loader, train_eval_loader, val_loader, history, fitted_pcas = run_training(train_dataset=train_dataset, val_dataset=val_dataset,
                                                                                          input_dims=domain_dims, shared_dim=shared_dim,
                                                                                          max_epochs = 100, patience = 25,SEED = SEED, **best_params)

with open(f"{model_output_dir}/fitted_pcas.pkl", "wb") as f:
    pickle.dump(fitted_pcas, f)

with open(f"{results_dir}/history.json", "w") as f:
    json.dump(history, f)

torch.save({
    "projectors_state_dict": trainer.projectors.state_dict(),
    "classifier_state_dict": trainer.classifier.state_dict(),
    "input_dims": input_dims,
    "shared_dim": shared_dim,
    "hidden_dim": best_params["hidden_dim"],
    "dropout": best_params["dropout"],
}, f"{model_output_dir}/best_model.pt")

#save outputs to files (plots, models, logits)

save_loss_curve(history, title="Train vs Val Loss", filename=f"plots/utterance{RUN}/loss_curve.png")

save_accuracy_curve(history["train_acc"], history["val_acc"], title="Train vs Val Accuracy", filename=f"plots/utterance{RUN}/accuracy_curve.png")
save_accuracy_curve(history["train_bal_acc"], history["val_bal_acc"], title="Train vs Val Balanced Accuracy", filename=f"plots/utterance{RUN}/bal_acc_curve.png")


save_split_results(trainer, train_eval_loader, "train", results_dir)
save_split_results(trainer, val_loader, "validation", results_dir)


with open(os.path.join(results_dir, "config.json"), "w") as f:
    json.dump({"run": RUN, "seed": SEED, "shared_dim": shared_dim,
               "max_epochs": 100, "patience": 25, **best_params}, f, indent=2)