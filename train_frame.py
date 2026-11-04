import torch
import random
import numpy as np
import os
import json
import argparse

from src.datasets_classes import MultiDomainDataset
from src.utils.evaluation import save_loss_curve, save_accuracy_curve
from src.utils.results import save_split_results
from src.utils.train_utils import run_training_frame
from src.config import NATIVE_MODEL_BY_DOMAIN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--bilstm", action="store_true")
parser.add_argument("--dann", action="store_true")
parser.add_argument("--sampler", action="store_true")
parser.add_argument("--alpha", type=float, default=1.0)
args = parser.parse_args()

name = "bilstm" if args.bilstm else "lstm"
if args.dann:
    name += f"_dann_a{args.alpha}"
if args.sampler:
    name += "_sampler"


train_dataset = MultiDomainDataset("outputs/embeddings/frame/train", native_only = True, w2v=True)
val_dataset = MultiDomainDataset("outputs/embeddings/frame/validation", native_only=True, w2v=True)

trainer, train_loader, train_eval_loader, val_loader, history, model = run_training_frame(train_dataset, val_dataset, lr=1e-4,
                                                                                          bilstm=args.bilstm, dann=args.dann,
                                                                                          use_sampler=args.sampler, alpha=args.alpha)

#save outputs to files (plots, models, logits)
plots_output_dir = os.path.join("outputs/plots/frame", name)
model_output_dir = os.path.join("outputs/models/frame", name)
results_dir = os.path.join("outputs/results/frame", name)
for d in (plots_output_dir, model_output_dir, results_dir):
    os.makedirs(d, exist_ok=True)

save_loss_curve(history, title = f"Loss curve ({name})", filename=f"plots/frame/{name}/loss_curve.png")
save_accuracy_curve(history["train_acc"], history["val_acc"], title=f"Accuracy curve ({name})", filename=f"plots/frame/{name}/accuracy_curve.png")
save_accuracy_curve(history["train_bal_acc"], history["val_bal_acc"], title=f"Balanced accuracy curve ({name})", filename=f"plots/frame/{name}/bal_accuracy_curve.png")

torch.save(model.state_dict(), os.path.join(model_output_dir, "best_model.pt"))
print(f"Model saved to {model_output_dir}")

save_split_results(trainer, train_eval_loader, "train", out_dir = results_dir) #save logits, labels, domains, loss of training set to a .pt file
save_split_results(trainer, val_loader, "validation", out_dir = results_dir) #save logits, labels, domains, loss of validation set to a .pt file

#torch.save(history, os.path.join(results_dir, "history.pt"))
with open(f"{results_dir}/history.json", "w") as f:
    json.dump(history, f)

with open(os.path.join(results_dir, "config.json"), "w") as f:
    json.dump({**vars(args), "seed": SEED}, f, indent=2)