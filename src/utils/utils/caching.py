import io
import numpy as np
import torch


def cache_embeddings(path, embeddings, labels, groups = None):
    torch.save({"embedding": embeddings, "label": labels, "group": groups}, path)


def load_embeddings(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(path, map_location=device)
    return data["embedding"], data["label"], data["group"]





