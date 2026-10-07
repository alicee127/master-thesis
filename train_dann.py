import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import os

from src.datasets_classes import frame_multidomain_collate_fn, MultiDomainDataset
from src.trainer_dann import TrainerDANN
from src.dann_pipeline import AudioEmotionDANN
from src.config import MODEL_OUTPUT_DIM_FRAME
from src.utils.evaluation import save_loss_curve, save_accuracy_curve
from src.utils.results import save_split_results

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

RUN = "frame_dann"

train_dataset = MultiDomainDataset("outputs/embeddings/frame/train", native_only = True)
val_dataset = MultiDomainDataset("outputs/embeddings/frame/validation", native_only=True)

input_dims = MODEL_OUTPUT_DIM_FRAME

#define the model
model = AudioEmotionDANN()

trainer = TrainerDANN(model, lr=0.001, alpha=1.0, domain_weight=1.0)

#create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size= 32, shuffle = True, collate_fn=frame_multidomain_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=frame_multidomain_collate_fn)

#train the model
history = trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=100, patience=10)

#save outputs to files (plots, models, logits)

plots_output_dir = os.path.join("outputs/plots", RUN)
model_output_dir = os.path.join("outputs/models", RUN)
results_output_dir = os.path.join("outputs/results", RUN)
for d in (plots_output_dir, model_output_dir, results_output_dir):
    os.makedirs(d, exist_ok=True)

save_loss_curve(history, title = "Loss curve with LSTM + DANN", filename=f"plots/{RUN}/loss_curve.png")
save_accuracy_curve(history["train_acc"], history["val_acc"], title="Accuracy curve with LSTM + DANN", filename=f"plots/{RUN}/accuracy_curve.png")
save_accuracy_curve(history["train_bal_acc"], history["val_bal_acc"], title="Balanced accuracy curve with LSTM + DANN", filename=f"plots/{RUN}/bal_accuracy_curve.png")

torch.save(model.state_dict(), os.path.join(model_output_dir, "best_model.pt"))
print(f"Model saved to {model_output_dir}")

save_split_results(trainer, train_loader, "train", out_dir = results_output_dir) #save logits, labels, domains, loss of training set to a .pt file
save_split_results(trainer, val_loader, "validation", out_dir = results_output_dir) #save logits, labels, domains, loss of validation set to a .pt file

torch.save(history, os.path.join(results_output_dir, "history.pt"))