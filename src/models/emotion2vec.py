import torch
from src.models.baseModel import BaseModel
from funasr import AutoModel
import soundfile as sf
import tempfile

class Emotion2VecModel(BaseModel):
    def __init__(self):
        super().__init__(target_sr = 16000, output_dim = 1024)
        self.model = AutoModel(model="iic/emotion2vec_plus_large", trust_remote_code = True, disable_update=True, disable_pbar=True, log_level="ERROR")
        self.freeze(self.model.model)
        self.model.model.to(self.device)

    def encode(self, waveform):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, waveform, samplerate=self.target_sr)
            res = self.model.generate(
                tmp.name,
                granularity="utterance",
                extract_embedding=True
                )
        embedding = res[0]["feats"]
        return torch.tensor(embedding)
