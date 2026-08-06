from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import json
import numpy as np
import random



from src.datasets_classes import MultiDomainDataset, multidomain_collate_fn
from src.projector import Projector
from src.classifier import Classifier
from src.trainer import Trainer
from src.utils.training_reports import save_loss_curve, compute_domain_accuracies, save_accuracy_table

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


train_dataset = MultiDomainDataset("outputs/embeddings/train")
val_dataset = MultiDomainDataset("outputs/embeddings/validation")

input_dims = {"speech": 1024, "music": 1024, "animal": 768, "soundscapes": 512}
shared_dim = 256

projectors = nn.ModuleDict({domain: Projector(input_dim=dim, shared_dim=shared_dim) for domain, dim in input_dims.items()})

classifier = Classifier(shared_dim=shared_dim, hidden_dim=64, dropout=0.3)

trainer = Trainer(projectors, classifier, shared_dim=shared_dim, lr=0.001)

sampler = trainer.create_weighted_sampler(train_dataset.domains, train_dataset.labels, alpha=1)
train_loader = DataLoader(train_dataset, sampler = sampler, batch_size= 32, collate_fn=multidomain_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=multidomain_collate_fn)


history = trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=100, patience=10, verbose=True)

with open("outputs/models/history.json", "w") as f:
    json.dump(history, f)

torch.save({
    "projectors_state_dict": trainer.projectors.state_dict(),
    "classifier_state_dict": trainer.classifier.state_dict(),
    "input_dims": input_dims,
    "shared_dim": shared_dim,
}, "outputs/models/best_model.pt")



save_loss_curve(history, title="Train vs Val Loss", filename="plots/loss_curve.png")

domains_accuracies = compute_domain_accuracies(trainer, train_loader, val_loader)
save_accuracy_table(domains_accuracies)
