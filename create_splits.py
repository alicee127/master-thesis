import torch
import os
import glob

from src.utils.splitting import train_val_test_split
from src.utils.caching import load_embeddings

EMBEDDINGS_DIR = "outputs/embeddings"
SPLIT_DIRS = {
    "train": os.path.join(EMBEDDINGS_DIR, "train"),
    "val": os.path.join(EMBEDDINGS_DIR, "validation"),
    "test": os.path.join(EMBEDDINGS_DIR, "test")
}

for split_dir in SPLIT_DIRS.values():
    os.makedirs(split_dir, exist_ok=True)

pt_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.pt"))

for filepath in pt_files:
    filename = os.path.basename(filepath)
    domain_name, _ = filename.split("_", 1)

    embeddings, labels, groups = load_embeddings(filepath)

    has_valid_groups = groups is not None and any(g is not None for g in groups)
    groups_to_use = groups if has_valid_groups else None

    (emb_train, labels_train, groups_train), (emb_val, labels_val, groups_val), (emb_test, labels_test, groups_test) = train_val_test_split(embeddings, labels, groups=groups_to_use)

    splits = {
        "train":  (emb_train, labels_train),
        "val": (emb_val, labels_val),
        "test": (emb_test, labels_test)
    }

    name_no_ext, ext = os.path.splitext(filename)

    for split_name, (emb, lab) in splits.items():
        out_filename = f"{name_no_ext}_{split_name}{ext}"
        out_path = os.path.join(SPLIT_DIRS[split_name], out_filename)
        torch.save({"embedding": emb, "label": lab}, out_path)
        print(f"Saved {split_name} split for '{domain_name}' -> {out_path} ({emb.shape[0]} samples)")

