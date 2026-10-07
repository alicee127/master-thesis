import torch
from src.models.baseModel import BaseModel
from transformers import ClapModel, ClapProcessor

class CLAPModel(BaseModel):
    def __init__(self, return_temporal = False):
        output_dim = 768 if return_temporal else 512 #CLAP's output dimension changes for frame- vs. utterance-level
        super().__init__(target_sr = 48000, output_dim= output_dim)

        self.return_temporal = return_temporal
        self.model = ClapModel.from_pretrained("laion/larger_clap_general")
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        self.freeze(self.model) #freeze the backbone to use it only for embedding extraction
        self.model.to(self.device) #use GPU or CPU depending on device settings

    def encode(self, waveform):
        inputs = self.processor(audio = waveform, sampling_rate = self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.return_temporal:
                outputs = self.model.get_audio_features(**inputs,  output_hidden_states=True)
                embedding = outputs.last_hidden_state # (1, sequence_length, 768)
            else:
                outputs = self.model.get_audio_features(**inputs)
                embedding = outputs.pooler_output # (1, 512)

        return embedding.squeeze(0)