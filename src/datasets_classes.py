import torch
from torch.utils.data import Dataset
import soundfile as sf
import io
import librosa
import pandas as pd
import os
import glob
import numpy as np
from datasets import load_dataset, get_dataset_split_names, concatenate_datasets, Audio
from src.utils.caching import load_embeddings


class BaseDataset(Dataset):
    def __init__(self, target_sr):
        self.target_sr = target_sr
        raw_data = self.load_data()
        self.data = self.preprocess_labels(raw_data)

        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:

        example = self.data[idx]

        waveform, sr = self.read_audio(example["audio"])
        waveform = self.to_mono(waveform)
        waveform = self.resampling(waveform, sr)
        waveform = self.pad_waveform(waveform)

        return waveform, example["label"]
    
    def load_data(self):
        #defined in each subclass
        raise NotImplementedError
    
    def preprocess_labels(self, data):
        #defined in each subclass
        raise NotImplementedError
    
    def resampling(self, waveform, original_sr):
        if original_sr == self.target_sr:
            return waveform

        return librosa.resample(waveform, orig_sr = original_sr, target_sr = self.target_sr)

    def to_mono(self, waveform):
        if waveform.ndim > 1:
            waveform = librosa.to_mono(waveform.T)
        return waveform
    
    def read_audio(self, audio_source):
        if isinstance(audio_source, bytes):
            return sf.read(io.BytesIO(audio_source))
        else:
            return sf.read(audio_source)


    def pad_waveform(self, waveform, min_duration=0.5):
        min_samples = int(min_duration * self.target_sr)

        if len(waveform) < min_samples:
            waveform = np.pad(waveform, (0, min_samples - len(waveform)), mode="constant")

        return waveform


class SpeechDataset(BaseDataset):
    '''Class for the speech dataset'''

    #define a mapping from categorical labels to binary labels
    #(happiness, excitement, encouragement, enthusiasm, calm) --> positive (1)
    #(anger, sadness, fear, disgust, anxiety, concern) --> negative (0)
    LABELS_MAP = {
    "happiness": 1,
    "excitement": 1,
    "encouragement": 1,
    "enthusiasm": 1,
    "calm": 1,
    "anger": 0,
    "sadness": 0,
    "fear": 0,
    "disgust": 0,
    "anxiety": 0,
    "concern": 0,
    }

    def __init__(self, target_sr = 16000, labels_map = None):
        self.labels_map = labels_map or self.LABELS_MAP
        super().__init__(target_sr)
    
    def load_data(self):
        splits = get_dataset_split_names("amu-cai/CAMEO")
        all_splits = [load_dataset("amu-cai/CAMEO", split = s) for s in splits]
        combined = concatenate_datasets(all_splits)
        combined = combined.cast_column("audio", Audio(decode=False))
        return combined
    
    def preprocess_labels(self, data):
        data = data.filter(lambda ex: ex["emotion"] in self.labels_map)
        data = data.map(lambda ex:{"label": self.labels_map[ex["emotion"]]})

        standardized = [
            {
                "audio": ex["audio"]["bytes"],
                "label": ex["label"],
                "group": ex["speaker_id"]
            }
            for ex in data
        ]

        return standardized


class MusicDataset(BaseDataset):

    CHORUS_PATH = "data/PMEmo2019/chorus"

    def __init__(self, target_sr = 24000):
        super().__init__(target_sr)
        #check that MERT sampling rate is actually 24kHz
    
    def load_data(self):
        annotations = pd.read_csv("data/PMEmo2019/annotations/static_annotations.csv")
        metadata = pd.read_csv("data/PMEmo2019/metadata.csv")

        merged = pd.merge(annotations, metadata, on = "musicId")
        
        return merged
    
    def preprocess_labels(self, data):
        standardized = []

        for _, row in data.iterrows():
            label = 1 if row["Valence(mean)"] >= 0.5 else 0

            file_path = os.path.join(self.CHORUS_PATH, row["fileName"])

            standardized.append({
                "audio": file_path,
                "label": label,
                "group": row["artist"] #for stratified splitting
            })


        return standardized


class AnimalVocDataset(BaseDataset):

    SOUNDS_PATH = "data/animal_vocalizations_dataset/SoundsDatabase"

    def __init__(self, target_sr = 16000):
        super().__init__(target_sr)

    
    def load_data(self):
        filenames = os.listdir(self.SOUNDS_PATH)
        return [os.path.join(self.SOUNDS_PATH, f) for f in filenames]
    
    def preprocess_labels(self, data):
        standardized = []

        for filepath in data:
            parsed = self.parse_filename(filepath)
            
            label = 1 if parsed["valence"] == "Positive" else 0
            
            standardized.append({
                "audio": filepath,
                "label": label,
                "group": parsed["animal_id"],
                "species": parsed["species"]
            })

        return standardized
    
    @staticmethod
    def parse_filename(filepath):
        filename = os.path.basename(filepath)
        name_no_ext = os.path.splitext(filename)[0]  #name without extension
        animal_id, species, context, valence, call_id = name_no_ext.split("-")

        return {
            "animal_id": animal_id, "species": species, "context": context, "valence": valence, "call_id": call_id
            }
    

class SoundscapesDataset(BaseDataset):

    AUDIO_ROOT = "data/Emo-Soundscapes/Emo-Soundscapes-Audio"
    VALENCE_CSV = "data/Emo-Soundscapes/Emo-Soundscapes-Ratings/Valence.csv"
    METADATA = "data/Emo-Soundscapes/Emo-Soundscapes-Metadata"

    def __init__(self, target_sr = 48000):
        #CLAP needs an input SR of 48kHz
        super().__init__(target_sr)
    
    def load_data(self):

        self.file_index = self.build_filename_index(self.AUDIO_ROOT)
        valence_df = pd.read_csv(self.VALENCE_CSV, header = None, names = ["FileName", "valence"])

        joined_files = os.path.join(self.METADATA, "*.csv")
        joined_list = glob.glob(joined_files)
        
        df_joined = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)

        df_joined["FileName"] = df_joined["FileName"].apply(self.fix_filename)

        merged = pd.merge(valence_df, df_joined, on=["FileName"], how = "left")
        
        return merged
    
    def preprocess_labels(self, data):

        standardized = []
        
        for _, row in data.iterrows():
            filename = row["FileName"]

            label = 1 if row["valence"] >=0 else 0

            file_path = self.file_index.get(row["FileName"])

            standardized.append({
                "audio": file_path,
                "label": label,
            })

        return standardized
    

    @staticmethod
    def build_filename_index(root_dir):
        index = {}

        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(".wav"):
                    index[f] = os.path.join(dirpath, f)
        return index
    
    @staticmethod
    def fix_filename(file_name):
        words = file_name.split(' ')
        file = None 
        for word in words:
            if '.' in word:
                file = word.split('.')

        new_name = file[0] + ".wav"

        return new_name



class MultiDomainDataset(Dataset):

    def __init__(self, embeddings_dir):
        self.domain_data = {}
        self.index_map = []

        pt_files = glob.glob(os.path.join(embeddings_dir, "*.pt"))

        for filepath in pt_files:
            filename = os.path.basename(filepath)
            domain_name, _ = filename.split("_", 1)

            data = torch.load(filepath)
            embeddings = data["embedding"]
            labels = data["label"]

            self.domain_data[domain_name] = {"embedding": embeddings, "label": labels}

            n_samples = embeddings.shape[0]
            self.index_map.extend([(domain_name, i) for i in range(n_samples)])

        self.domains = [d for d, _ in self.index_map]

        for domain_name in self.domain_data:
            labels = self.domain_data[domain_name]["label"]
            self.domain_data[domain_name]["label"] = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)

        self.labels = [self.domain_data[d]["label"][i] for d, i in self.index_map]


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        domain_name, local_idx = self.index_map[idx]
        embedding = self.domain_data[domain_name]["embedding"][local_idx]
        label = self.domain_data[domain_name]["label"][local_idx]

        return embedding, label, domain_name


def multidomain_collate_fn(batch):
    embeddings, labels, domains = zip(*batch)
    labels = torch.tensor(labels)
    return list(embeddings), labels, list(domains)