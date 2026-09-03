from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import copy
from sklearn.decomposition import PCA

from src.datasets_classes import multidomain_collate_fn
from src.classifier import Classifier
from src.trainer import Trainer

def run_training(train_dataset, val_dataset, input_dims, shared_dim, hidden_dim, lr, dropout, max_epochs = 100, patience = 10, SEED = 42):

    train_dataset = copy.deepcopy(train_dataset)
    val_dataset = copy.deepcopy(val_dataset)

    #apply PCA for dimensionality reduction (shared embedding space)
    fitted_pcas = {}

    for domain_name in train_dataset.domain_data:
        train_embeddings = train_dataset.domain_data[domain_name]["embedding"]
        if isinstance(train_embeddings, torch.Tensor):
            train_embeddings.numpy()

        pca = PCA(n_components=shared_dim, random_state=SEED)
        train_reduced = pca.fit_transform(train_embeddings)
        train_dataset.domain_data[domain_name]["embedding"] = torch.tensor(train_reduced, dtype=torch.float32)

        fitted_pcas[domain_name] = pca

        val_embeddings = val_dataset.domain_data[domain_name]["embedding"]
        if isinstance(val_embeddings, torch.Tensor):
            val_embeddings.numpy()

        val_reduced = pca.transform(val_embeddings)
        val_dataset.domain_data[domain_name]["embedding"] = torch.tensor(val_reduced, dtype=torch.float32)

    projectors = nn.ModuleDict({domain: nn.Identity() for domain in input_dims})

    classifier = Classifier(shared_dim=shared_dim, hidden_dim=hidden_dim, dropout=dropout)

    trainer = Trainer(projectors, classifier, shared_dim=shared_dim, lr=lr)

    sampler = trainer.create_weighted_sampler(train_dataset.domains, train_dataset.labels, alpha=1)
    train_loader = DataLoader(train_dataset, sampler = sampler, batch_size= 32, collate_fn=multidomain_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=multidomain_collate_fn)


    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=max_epochs, patience=patience, verbose=False)

    return trainer, train_loader, val_loader, history, fitted_pcas
