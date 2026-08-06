from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import numpy as np
import torch

def train_val_test_split(embeddings, labels, groups = None, n_splits = 10, test_split = 1, random_state = 42):

    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

    if groups is not None:
        groups_np = np.array(groups)

        sgkf_test = StratifiedGroupKFold(n_splits = n_splits, shuffle = True, random_state = random_state)
        train_val_idx, test_idx = next(sgkf_test.split(embeddings, labels_np, groups_np))

        emb_train_val, labels_train_val, groups_train_val = embeddings[train_val_idx], labels[train_val_idx], groups_np[train_val_idx]
        emb_test, labels_test, groups_test = embeddings[test_idx], labels[test_idx], groups_np[test_idx]

        groups_tv_np = np.array(groups_train_val)
        labels_tv_np = labels_train_val.cpu().numpy() if isinstance(labels_train_val, torch.Tensor) else np.array(labels_train_val)

        sgkf_val = StratifiedGroupKFold(n_splits = n_splits-test_split, shuffle=True, random_state= random_state)

        train_idx, val_idx = next(sgkf_val.split(emb_train_val, labels_tv_np, groups_tv_np))

        emb_train, labels_train, groups_train = emb_train_val[train_idx], labels_train_val[train_idx], groups_train_val[train_idx]
        emb_val, labels_val, groups_val = emb_train_val[val_idx], labels_train_val[val_idx], groups_train_val[val_idx]

    else:
        skf_test = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        train_val_idx, test_idx = next(skf_test.split(embeddings, labels_np))

        emb_train_val, labels_train_val = embeddings[train_val_idx], labels[train_val_idx]
        emb_test, labels_test = embeddings[test_idx], labels[test_idx]

        labels_tv_np = labels_train_val.cpu().numpy() if isinstance(labels_train_val, torch.Tensor) else np.array(labels_train_val)

        skf_val = StratifiedKFold(n_splits=n_splits-test_split, shuffle=True, random_state=random_state)
        train_idx, val_idx = next(skf_val.split(emb_train_val, labels_tv_np))

        emb_train, labels_train = emb_train_val[train_idx], labels_train_val[train_idx]
        emb_val, labels_val = emb_train_val[val_idx], labels_train_val[val_idx]

        groups_train, groups_val, groups_test = None, None, None

    return (
        (emb_train, labels_train, groups_train),
        (emb_val, labels_val, groups_val),
        (emb_test, labels_test, groups_test)
    )

    