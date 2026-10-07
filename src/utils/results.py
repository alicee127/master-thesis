import torch
import os

def save_split_results(trainer, loader, split_name: str, out_dir="outputs/results/frame"):
    os.makedirs(out_dir, exist_ok=True)

    loss, logits, labels, domains = trainer.evaluate(loader)
    path = os.path.join(out_dir, f"{split_name}_results.pt")

    torch.save({"logits": logits, "labels": labels, "domains": domains, "loss": loss}, path)

    print(f"Saved {split_name} results in {path}")