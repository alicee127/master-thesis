import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np
import copy
from tqdm.auto import tqdm

class Trainer:

    def __init__(self, projectors, classifier, shared_dim, lr = 0.001, projector_weight_decay=0.0, classifier_weight_decay=0.0):
        self.projectors = projectors
        self.classifier = classifier
        self.shared_dim = shared_dim

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.projectors.to(self.device)
        self.classifier.to(self.device)

        #loss function: BCE with logits
        self.criterion = nn.BCEWithLogitsLoss()

        #optimization with Adam optimizer
        #params = list(self.projectors.parameters()) + list(self.classifier.parameters())

        param_groups = [{"params": self.projectors.parameters(), "weight_decay": projector_weight_decay},
                        {"params": self.classifier.parameters(), "weight_decay": classifier_weight_decay}]
        self.optimizer = torch.optim.Adam(param_groups, lr)


    def create_weighted_sampler(self, domains, labels, alpha = 0.5):
        combos = [f"{d}_{l.item() if isinstance(l, torch.Tensor) else l}" for d, l in zip(domains,labels)]

        unique_combos, counts = np.unique(combos, return_counts=True)
        combo_counts = dict(zip(unique_combos, counts))

        sample_weights = [1.0 / combo_counts[c]**alpha for c in combos]

        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)

        return sampler


    def _forward_batch(self, embeddings: list, domains: list):
        """Routes each sample to its domain-specific projector"""
        domains_arr = np.array(domains)
        projected = torch.empty(len(embeddings), self.shared_dim, device=self.device)

        for domain_name in np.unique(domains_arr):

            idx = np.where(domains_arr == domain_name)[0]
            domain_embeddings = torch.stack([embeddings[i] for i in idx]).to(self.device)

            projected[idx] = self.projectors[domain_name](domain_embeddings)

        return projected

    def _run_epoch(self, data_loader: DataLoader, train: bool):
        self.projectors.train(train)
        self.classifier.train(train)

        running_loss = 0.0
        all_logits, all_labels = [], []

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for embeddings, labels, domains in data_loader:
                labels = labels.to(self.device).float().unsqueeze(1)

                if train:
                    self.optimizer.zero_grad()

                projected = self._forward_batch(embeddings, domains)
                logits = self.classifier(projected)
                loss = self.criterion(logits, labels)

                if train:
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * len(embeddings)
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        epoch_loss = running_loss / len(data_loader.dataset)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return epoch_loss, all_logits, all_labels

    def train_one_epoch(self, train_loader):
        epoch_loss, _, _ = self._run_epoch(train_loader, train=True)
        return epoch_loss


    def evaluate(self, data_loader):
        epoch_loss, all_logits, all_labels = self._run_epoch(data_loader, train=False)
        return epoch_loss, all_logits, all_labels

    def fit(self, train_loader, val_loader, max_epochs = 1000, patience = 10, verbose = True, min_delta = 0.001):
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        history = {"train_loss": [], "val_loss": []}

        epoch_iterator  = tqdm(range(max_epochs), desc = "Training")

        for epoch in epoch_iterator:
            train_loss = self.train_one_epoch(train_loader)
            val_loss, _, _ = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)


            epoch_iterator.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

            if verbose:
                print(f"Epoch {epoch+1}/{max_epochs} "
                      f"- train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = {"projectors_state_dict": copy.deepcopy(self.projectors.state_dict()),
                              "classifier_state_dict": copy.deepcopy(self.classifier.state_dict())}

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > patience:
                if verbose:
                    print(f"Early stopping after epoch {epoch+1}\n(no improvement for {patience} epochs)")
                epoch_iterator.close()
                break

        if best_state is not None:
            self.projectors.load_state_dict(best_state["projectors_state_dict"])
            self.classifier.load_state_dict(best_state["classifier_state_dict"])

        return history
    

    

