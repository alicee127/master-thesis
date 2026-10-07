import torch
import torch.nn as nn
from tqdm.auto import tqdm
import copy
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.config import DEVICE, NATIVE_MODEL_BY_DOMAIN


class TrainerLSTM:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.device = DEVICE
        self.model.to(self.device)

        #loss function: BCE with logits
        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, "min")

    def _run_epoch(self, data_loader, train: bool):

        self.model.train(train)

        running_loss = 0.0
        all_logits, all_labels, all_domains = [], [], []

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for embeddings, labels, domains in data_loader:
                
                embeddings = [e.to(self.device) for e in embeddings]
                model_names = [NATIVE_MODEL_BY_DOMAIN[d] for d in domains]
                labels = labels.to(self.device).float()
        
                if train:
                    self.optimizer.zero_grad()
        
                logits, _ = self.model(embeddings, model_names)
                loss = self.criterion(logits, labels)
        
                if train:
                    loss.backward()
                    self.optimizer.step()
        
                running_loss += loss.item() * len(labels)
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_domains.extend(domains)
        
        epoch_loss = running_loss / len(data_loader.dataset)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return epoch_loss, all_logits, all_labels, all_domains

    def train_one_epoch(self, train_loader):
        return self._run_epoch(train_loader, train=True)

    def evaluate(self, data_loader):
        """Returns loss, logits, labels, domains"""
        return self._run_epoch(data_loader, train=False)

    def fit(self, train_loader, val_loader, max_epochs = 1000, patience = 10, min_delta = 0.001, checkpoint_path = None, resume_from = None):
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_bal_acc": [], "val_bal_acc": []}

        start_epoch = 0
        if resume_from is not None:
            ckpt = self._load_checkpoint(resume_from)
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            epochs_without_improvement = ckpt["epochs_without_improvement"]
            best_state = ckpt["best_state"]
            history = ckpt["history"]

        epoch_iterator  = tqdm(range(start_epoch, max_epochs), desc="Training")

        for epoch in epoch_iterator:
            train_loss, train_logits, train_labels, train_domains = self.train_one_epoch(train_loader)
            val_loss, val_logits, val_labels, val_domains = self.evaluate(val_loader)

            train_preds = (torch.sigmoid(train_logits) > 0.5).long()
            val_preds = (torch.sigmoid(val_logits) > 0.5).long()

            train_acc = accuracy_score(train_labels.long().numpy(), train_preds.numpy())
            val_acc = accuracy_score(val_labels.long().numpy(), val_preds.numpy())

            train_bal_acc = balanced_accuracy_score(train_labels.long().numpy(), train_preds.numpy())
            val_bal_acc = balanced_accuracy_score(val_labels.long().numpy(), val_preds.numpy())

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_bal_acc"].append(train_bal_acc)
            history["val_bal_acc"].append(val_bal_acc)


            epoch_iterator.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")


            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1

            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, epoch, best_val_loss, epochs_without_improvement, best_state, history)


            if epochs_without_improvement >= patience:
                epoch_iterator.close()
                break

            #self.scheduler.step(val_loss)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def _save_checkpoint(self, path, epoch, best_val_loss, epochs_without_improvement, best_state, history):
        """Saves the checkpoint after each epoch, to recover training in case of undesired interruptions"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "best_val_loss": best_val_loss,
            "best_state": best_state,
            "history": history
        }

        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def _load_checkpoint(self, path):
        """Loads the last saved checkpoint to resume training in case of undesired interruptions"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint



class TrainerDANN(TrainerLSTM):
    def __init__(self, model, lr=0.001, alpha = 1.0, domain_weight=1.0):
        super().__init__(model, lr)

        self.alpha = alpha
        self.domain_weight = domain_weight
        self.domain_to_idx = {d: i for i, d in enumerate(NATIVE_MODEL_BY_DOMAIN)}

        self.domain_criterion = nn.CrossEntropyLoss() #loss function for the domain discriminator
        self.domain_loss_history = []

    def _run_epoch(self, data_loader, train: bool):
    
        self.model.train(train)
    
        running_loss = 0.0
        running_domain = 0.0
        all_logits, all_labels, all_domains = [], [], []
    
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for embeddings, labels, domains in data_loader:

                embeddings = [e.to(self.device) for e in embeddings]
                model_names = [NATIVE_MODEL_BY_DOMAIN[d] for d in domains]
                labels = labels.to(self.device).float()
    
                if train:
                    self.optimizer.zero_grad()
                    logits, domain_logits, _ = self.model(embeddings, model_names, alpha=self.alpha)
                    domain_labels = torch.tensor([self.domain_to_idx[d] for d in domains], device=self.device)

                    #compute loss as emotion classifier loss + domain weight * domain discriminator loss (negative)
                    #loss = self.criterion(logits, labels) + self.domain_weight * self.domain_criterion(domain_logits, domain_labels)
                    emotion_loss = self.criterion(logits, labels)
                    domain_loss = self.domain_criterion(domain_logits, domain_labels)
                    loss = emotion_loss + self.domain_weight * domain_loss
                    loss.backward()
                    self.optimizer.step()

                    running_domain += domain_loss.item() * len(labels)
                else:
                    logits, _, _ = self.model(embeddings, model_names)
                    emotion_loss = self.criterion(logits, labels)
                    
    
                running_loss += emotion_loss.item() * len(labels)
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_domains.extend(domains)
    
        epoch_loss = running_loss / len(data_loader.dataset)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if train:
            self.domain_loss_history.append(running_domain / len(data_loader.dataset))
    
        return epoch_loss, all_logits, all_labels, all_domains
