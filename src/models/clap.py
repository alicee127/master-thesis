import torch
from src.models.baseModel import BaseModel
from transformers import ClapModel, ClapProcessor

class CLAPModel(BaseModel):
    def __init__(self):
        super().__init__(target_sr = 48000, output_dim = 512)
        self.model = ClapModel.from_pretrained("laion/larger_clap_general")
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        self.freeze(self.model) #freeze the backbone to use it only for embedding extraction
        self.model.to(self.device) #use GPU or CPU depending on device settings

    def encode(self, waveform):
        inputs = self.processor(audio = waveform, sampling_rate = self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_audio_features(**inputs)
            embedding = outputs.pooler_output

        return embedding.squeeze(0)