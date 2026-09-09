import argparse
import os
import torch
from src.datasets_classes import SpeechDataset, MusicDataset, AnimalVocDataset, SoundscapesDataset
from src.utils.caching import cache_embeddings
from src.config import MODEL_SR

#read domain name and model from command line
parser = argparse.ArgumentParser()
parser.add_argument("domain", choices=["speech", "music", "animal", "soundscapes"], help="Which domain's dataset to use")
parser.add_argument("model", choices=["emotion2vec", "mert", "aves2", "clap"], help="Which foundation model to use")
args = parser.parse_args()
domain_name = args.domain
model_name = args.model



#create outputs/embeddings path
os.makedirs("outputs/embeddings", exist_ok = True)


CHUNK_SIZE = 500 #save progress every 500 samples

#pick dataset based on domain

if domain_name == "speech":
    dataset = SpeechDataset(target_sr=MODEL_SR[model_name])

elif domain_name == "music":
    dataset = MusicDataset(target_sr=MODEL_SR[model_name])

elif domain_name == "animal":
    dataset = AnimalVocDataset(target_sr=MODEL_SR[model_name])
    
elif domain_name == "soundscapes":
    dataset = SoundscapesDataset(target_sr=MODEL_SR[model_name])


#import the model class inside an if-else pile to avoid dependencies conflicts (MERT vs funasr)

if model_name == "emotion2vec":
    from src.models.emotion2vec import Emotion2VecModel
    model = Emotion2VecModel()
    model_tag = "e2v"

elif model_name == "mert":
    from src.models.mert import MERTModel
    model = MERTModel()
    model_tag = "mert"

elif model_name == "aves2":
    from src.models.aves2 import AVES2Model
    model = AVES2Model()
    model_tag = "aves2"

elif model_name == "clap":
    from src.models.clap import CLAPModel
    model = CLAPModel()
    model_tag = "clap"

output_path = f"outputs/embeddings/{domain_name}_{model_tag}.pt"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

if os.path.exists(output_path):
    checkpoint = torch.load(output_path)
    all_embeddings = list(checkpoint["embedding"])
    all_labels = list(checkpoint["label"])
    all_groups = checkpoint["group"]
    start_idx = len(all_embeddings)
    print(f"Resuming from sample {start_idx}")
else:
    all_embeddings = []
    all_labels = []
    all_groups = []
    start_idx = 0


for i in range(start_idx, len(dataset)):
    waveform, label = dataset[i]
    embedding = model.encode(waveform)
    all_embeddings.append(embedding)
    all_labels.append(label)
    all_groups.append(dataset.data[i].get("group"))

    if i % 100 == 0:
        print(f"Processed {i}/{len(dataset)}")

    if (i + 1) % CHUNK_SIZE == 0 or i == len(dataset) - 1:
        embeddings_tensor = torch.stack(all_embeddings)
        labels_tensor = torch.tensor(all_labels)
        cache_embeddings(output_path, embeddings_tensor, labels_tensor, all_groups)
        print(f"Checkpoint saved at sample {i+1}/{len(dataset)}")
        
cache_embeddings(output_path, embeddings_tensor, labels_tensor, all_groups)
print(f"Saved {len(dataset)} embeddings to {output_path}")
