import os
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader

from src.classifier import Classifier
from src.trainer import Trainer
from src.datasets_classes import MultiDomainDataset, multidomain_collate_fn, frame_multidomain_collate_fn
from src.utils.results import save_split_results
from src.dann_pipeline import AudioEmotionDANN
from src.trainer_dann import TrainerLSTM, TrainerDANN
from src.config import DEVICE

APPROACHES = [
    {"name": "utterance", "level": "utterance", "w2v": False, "native_only": True, "model": "outputs/models/utterance/best_model.pt", "pca": "outputs/models/utterance/fitted_pcas.pkl"},
    {"name": "utterance_w2v", "level": "utterance", "w2v": True, "native_only": True, "model": "outputs/models/utterance_w2v/best_model.pt", "pca": "outputs/models/utterance_w2v/fitted_pcas.pkl"},
    {"name": "e2v_only", "level": "utterance", "model_filter":"e2v", "model": "outputs/models/utterance_single/e2v/best_model.pt"},
    {"name": "mert_only", "level": "utterance", "model_filter":"mert", "model": "outputs/models/utterance_single/mert/best_model.pt"},
    {"name": "aves2_only", "level": "utterance", "model_filter":"aves2", "model": "outputs/models/utterance_single/aves2/best_model.pt"},
    {"name": "clap_only", "level": "utterance", "model_filter":"clap", "model": "outputs/models/utterance_single/clap/best_model.pt"},
    {"name": "lstm", "level": "frame", "bilstm": False, "dann": False, "model": "outputs/models/frame/lstm/best_model.pt"},
    {"name": "bilstm", "level": "frame", "bilstm": True, "dann": False, "model": "outputs/models/frame/bilstm/best_model.pt"},
    {"name": "lstm_dann_a0.2", "level": "frame", "bilstm": False, "dann": True, "alpha": 0.2,  "model": "outputs/models/frame/lstm_dann_a0.2/best_model.pt"},
    {"name": "bilstm_dann_a0.3", "level": "frame", "bilstm": True, "dann": True, "alpha": 0.3, "model": "outputs/models/frame/bilstm_dann_a0.3/best_model.pt"}
]


def apply_pca(dataset, pca_path):
    with open(pca_path, "rb") as f:
        fitted_pcas = pickle.load(f)
    
    for domain_name, d in dataset.domain_data.items():
        emb = d["embedding"]
        if isinstance(emb, torch.Tensor):
           emb = emb.numpy()

        reduced = fitted_pcas[domain_name].transform(emb)
        d["embedding"] = torch.tensor(reduced, dtype=torch.float32)

def load_utterance_trainer(model_path, domains):

    checkpoint = torch.load(model_path, map_location=DEVICE)
    projectors = nn.ModuleDict({d: nn.Identity() for d in domains})
    classifier = Classifier(shared_dim=checkpoint["shared_dim"], hidden_dim = checkpoint["hidden_dim"], dropout=checkpoint["dropout"])

    projectors.load_state_dict(checkpoint["projectors_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    projectors.to(DEVICE).eval()
    classifier.to(DEVICE).eval()

    trainer = Trainer(projectors, classifier, shared_dim=checkpoint["shared_dim"])

    return trainer


for a in APPROACHES:
    results_dir = os.path.dirname(a["model"]).replace("outputs/models", "outputs/results")
    os.makedirs(results_dir, exist_ok=True)

    if a["level"] == "frame":
        test_dataset = MultiDomainDataset("outputs/embeddings/frame/test", native_only=True, w2v=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=frame_multidomain_collate_fn)


        model = AudioEmotionDANN(bilstm = a["bilstm"], dann = a["dann"]).to(DEVICE)
        model.load_state_dict(torch.load(a["model"], map_location=DEVICE))
        model.eval()


        if a["dann"]:
            trainer = TrainerDANN(model, lr=1e-4, alpha=a["alpha"])
        else:
            trainer = TrainerLSTM(model, lr=1e-4)

        save_split_results(trainer, test_loader, "test", out_dir=results_dir)
    
    elif a["level"] == "utterance":
        test_dataset = MultiDomainDataset("outputs/embeddings/utterance/test", model_filter=a.get("model_filter"), native_only=a.get("native_only", False), w2v=a.get("w2v", False))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=multidomain_collate_fn)

        

        if "pca" in a:
            apply_pca(test_dataset, a["pca"])

        trainer = load_utterance_trainer(a["model"], test_dataset.domain_data.keys())
        
        save_split_results(trainer, test_loader, "test",  out_dir = results_dir)