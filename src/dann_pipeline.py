import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from src.config import DEVICE, MODEL_OUTPUT_DIM_FRAME
from src.classifier import Classifier
from src.discriminator import Discriminator, GradientReversalLayer

class AudioEmotionDANN(nn.Module):

    def __init__(self, bilstm = False, dann = False):
        super().__init__()

        self.input_dims = MODEL_OUTPUT_DIM_FRAME
        self.shared_dim = 256
        self.hidden_dim = 32 if bilstm else 64
        self.num_domains = 4
        self.bilstm = bilstm
        self.dann = dann
        self.out_dim = self.hidden_dim*2 if bilstm else self.hidden_dim

        self.projectors = nn.ModuleDict({
            "e2v": nn.Linear(self.input_dims["e2v"], self.shared_dim),
            "mert": nn.Linear(self.input_dims["mert"], self.shared_dim),
            "aves2": nn.Linear(self.input_dims["aves2"], self.shared_dim),
            "wav2vec": nn.Linear(self.input_dims["wav2vec"], self.shared_dim)
        })


        #shared LSTM component (change bidirectional to True for BiLSTM)
        self.shared_lstm = nn.LSTM(input_size=self.shared_dim, hidden_size=self.hidden_dim, batch_first=True, bidirectional=self.bilstm)

        #simple MLP for binary emotion classification
        self.emotion_classifier = Classifier(shared_dim=self.out_dim, hidden_dim=64, dropout = 0.3)

        #adversarial MLP for domain discrimination (DANN)
        self.grl = GradientReversalLayer()
        self.domain_discriminator = Discriminator(input_dim=self.out_dim, hidden_dim=64, num_domains=self.num_domains)

    def forward(self, x, model_names, alpha = 1.0):
        lengths = torch.tensor([s.size(0) for s in x])
        x_shared = [self.projectors[n](s) for s, n in zip(x, model_names)]

        x_padded = pad_sequence(x_shared, batch_first=True)
        packed = pack_padded_sequence(x_padded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        lstm_out, (hidden, cell) = self.shared_lstm(packed)
        if self.bilstm:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last_hidden = hidden[-1]

        emotion_logits = self.emotion_classifier(last_hidden).squeeze(-1)

        if self.training and self.dann:
            domain_logits = self.domain_discriminator(self.grl(last_hidden, alpha))
            return emotion_logits, domain_logits, last_hidden

        return emotion_logits, None, last_hidden