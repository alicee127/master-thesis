import torch
from src.models.baseModel import BaseModel
from transformers import AutoModel, Wav2Vec2FeatureExtractor

class MERTModel(BaseModel):
    def __init__(self):
        super().__init__(target_sr = 24000, output_dim = 1024)
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        self.freeze(self.model) #freeze the backbone to use it only for embedding extraction
        self.model.to(self.device) #use GPU or CPU depending on device settings

    def encode(self, audio):

        inputs = self.processor(audio, sampling_rate = self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze(0)