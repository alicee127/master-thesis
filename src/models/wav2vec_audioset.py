import torch
from src.models.baseModel import BaseModel
from transformers import AutoModel, AutoFeatureExtractor

class W2VAudioSet(BaseModel):
    def __init__(self, return_temporal = False):
        super().__init__(target_sr = 16000, output_dim = 768)

        self.return_temporal = return_temporal
        self.model = AutoModel.from_pretrained("ALM/wav2vec2-base-audioset")
        self.processor = AutoFeatureExtractor.from_pretrained("ALM/wav2vec2-base-audioset")
        self.freeze(self.model) #freeze the backbone to use it only for embedding extraction
        self.model.to(self.device) #use GPU or CPU depending on device settings

    def encode(self, waveform):
        inputs = self.processor(waveform, sampling_rate = self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            if self.return_temporal:
                embedding = outputs.last_hidden_state # (1, T, 768)
            else:
                embedding = outputs.last_hidden_state.mean(dim=1) # (1, 768)

        return embedding.squeeze(0)