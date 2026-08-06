import torch
from src.models.baseModel import BaseModel
import avex

class AVES2Model(BaseModel):
    def __init__(self):
        super().__init__(target_sr = 16000, output_dim = 768)
        self.model = avex.load_model("esp_aves2_sl_beats_bio", device=self.device, return_features_only=True)
        self.freeze(self.model)
        self.model.to(self.device)

    def encode(self, waveform):
        with torch.no_grad():
            audio_tensor = torch.tensor(waveform).unsqueeze(0).to(self.device)
            embeddings = self.model(audio_tensor)
            embedding = embeddings.mean(dim = 1)

        return embedding.squeeze(0)