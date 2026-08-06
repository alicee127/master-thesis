import argparse
import os
import torch
from src.datasets_classes import SpeechDataset, MusicDataset, AnimalVocDataset, SoundscapesDataset
from src.utils.caching import cache_embeddings

#read domain name from command line
parser = argparse.ArgumentParser()
parser.add_argument("domain", choices=["speech", "music", "animal", "soundscapes"])
args = parser.parse_args()
domain_name = args.domain

#create outputs/embeddings path
os.makedirs("outputs/embeddings", exist_ok = True)


CHUNK_SIZE = 500 #save progress every 500 samples

#import the model class inside an if-else pile to avoid dependencies conflicts (MERT vs funasr)

if domain_name == "speech":
    from src.models.emotion2vec import Emotion2VecModel
    dataset = SpeechDataset()
    model = Emotion2VecModel()
    output_path = "outputs/embeddings/speech_e2v.pt"

elif domain_name == "music":
    from src.models.mert import MERTModel
    dataset = MusicDataset()
    model = MERTModel()
    output_path = "outputs/embeddings/music_mert.pt"

elif domain_name == "animal":
    from src.models.aves2 import AVES2Model
    dataset = AnimalVocDataset()
    model = AVES2Model()
    output_path = "outputs/embeddings/animal_aves2.pt"

elif domain_name == "soundscapes":
    from src.models.clap import CLAPModel
    dataset = SoundscapesDataset()
    model = CLAPModel()
    output_path = "outputs/embeddings/soundscapes_clap.pt"


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

#embeddings_tensor = torch.stack(all_embeddings)
#labels_tensor = torch.tensor(all_labels)

cache_embeddings(output_path, embeddings_tensor, labels_tensor, all_groups)
print(f"Saved {len(dataset)} embeddings to {output_path}")
