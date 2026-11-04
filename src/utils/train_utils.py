from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch
import numpy as np
import copy
from sklearn.decomposition import PCA

from src.datasets_classes import multidomain_collate_fn, frame_multidomain_collate_fn
from src.classifier import Classifier
from src.trainer import Trainer
from src.trainer_dann import TrainerLSTM, TrainerDANN
from src.dann_pipeline import AudioEmotionDANN

def create_weighted_sampler(domains, labels):
    """Weighted sampler that balances every (domain, label) combination.
    
    Each sample gets weight 1/(count of its domain + label combination), sampled with replacement, small domains/classes are oversampled.
    """
    combos = [f"{d}_{l.item() if isinstance(l, torch.Tensor) else l}" for d, l in zip(domains,labels)]

    unique_combos, counts = np.unique(combos, return_counts=True)
    combo_counts = dict(zip(unique_combos, counts))
    sample_weights = [1.0 / combo_counts[c] for c in combos]

    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

    return sampler



def run_training(train_dataset, val_dataset, input_dims, shared_dim, hidden_dim, lr, dropout, max_epochs = 100, patience = 10, SEED = 42):
    """Train the utterance-level PCA + MLP pipeline.

    Fits one PCA per domain on the training embeddings (to shared_dim components) and applies it to validation.
    Train loader uses a weighted sampler.

    Returns: trainer, train_loader, val_loader, history, fitted_pcas  
    """

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

    sampler = create_weighted_sampler(train_dataset.domains, train_dataset.labels)
    train_loader = DataLoader(train_dataset, sampler = sampler, batch_size= 32, collate_fn=multidomain_collate_fn)
    train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=frame_multidomain_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=multidomain_collate_fn)


    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=max_epochs, patience=patience, verbose=False)

    return trainer, train_loader, train_eval_loader, val_loader, history, fitted_pcas



def run_training_no_pca(train_dataset, val_dataset, shared_dim, hidden_dim, lr, dropout, max_epochs = 100, patience = 10, SEED = 42):
    """Train the utterance-level MLP pipeline, embeddings are used as they are (no PCA).

    Used for the single foundation model experiments.
    
    Returns: trainer, train_loader, val_loader, history  
    """
    train_dataset = copy.deepcopy(train_dataset)
    val_dataset = copy.deepcopy(val_dataset)

    domains = list(train_dataset.domain_data.keys())
    projectors = nn.ModuleDict({domain: nn.Identity() for domain in domains})

    classifier = Classifier(shared_dim=shared_dim, hidden_dim=hidden_dim, dropout=dropout)
    trainer = Trainer(projectors, classifier, shared_dim=shared_dim, lr=lr)

    sampler = create_weighted_sampler(train_dataset.domains, train_dataset.labels)
    train_loader = DataLoader(train_dataset, sampler = sampler, batch_size= 32, collate_fn=multidomain_collate_fn)
    train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=frame_multidomain_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=multidomain_collate_fn)


    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=max_epochs, patience=patience, verbose=False)

    return trainer, train_loader, train_eval_loader, val_loader, history


def run_training_frame(train_dataset, val_dataset, lr=1e-3, bilstm=False, dann=False, use_sampler=False, alpha=1.0, max_epochs=100, patience=10):
    """Train a frame-level LSTM/BiLSTM + MLP model, with a Linear Projector and optionally with a DANN domain discriminator.

    If use_sampler is True: train loader uses a weighted sampler balancing (domain, label) combinations.
    alpha: gradient reversal strenght for DANN, only used when dann = True

    Returns: trainer, train_loader, train_eval_loader, val_loader, history, model
    """

    train_dataset = copy.deepcopy(train_dataset)
    val_dataset = copy.deepcopy(val_dataset)

    model = AudioEmotionDANN(bilstm, dann)

    #apply weighted sampler to training data if use_sampler is True
    if use_sampler:
        sampler = create_weighted_sampler(train_dataset.domains, train_dataset.labels)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, collate_fn=frame_multidomain_collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size= 32, shuffle = True, collate_fn=frame_multidomain_collate_fn)

    train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=frame_multidomain_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=frame_multidomain_collate_fn)

    if dann:
        trainer = TrainerDANN(model, lr, alpha)
    else:
        trainer = TrainerLSTM(model, lr)

    history = trainer.fit(train_loader, val_loader, max_epochs, patience)

    return trainer, train_loader, train_eval_loader, val_loader, history, model
