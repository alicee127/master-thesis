import torch
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self, target_sr, output_dim):
        super().__init__()
        self.target_sr = target_sr
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def encode(self):
        #define in each subclass
        raise NotImplementedError

'''class Emotion2VecModel(BaseModel):
    def __init__(self):
        super().__init__(target_sr = 16000, output_dim = 1024)
        self.model = funasr.AutoModel(model="iic/emotion2vec_plus_large", trust_remote_code = True)
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
        return torch.tensor(embedding)'''

'''class MERTModel(BaseModel):
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
        return embedding.squeeze(0)'''
    
'''class AVES2Model(BaseModel):
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
    '''


'''class CLAPModel(BaseModel):
    def __init__(self):
        super().__init__(target_sr = 48000, output_dim = 512)
        self.model = ClapModel.from_pretrained("laion/larger_clap_general")
        self.processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
        self.freeze(self.model) #freeze the backbone to use it only for embedding extraction
        self.model.to(self.device) #use GPU or CPU depending on device settings

    def encode(self, waveform):
        inputs = self.processor(waveform, sampling_rate = self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_audio_features(**inputs)
            embedding = outputs.pooler_output

        return embedding.squeeze(0)'''
    