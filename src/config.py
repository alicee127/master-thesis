MODEL_SR = {
    "emotion2vec": 16000,
    "mert": 24000,
    "aves2": 16000,
    "clap": 48000
    }

NATIVE_MODEL_BY_DOMAIN = {
    "speech": "e2v",
    "music": "mert",
    "animal": "aves2",
    "soundscapes": "clap"
}

MODEL_OUTPUT_DIM = {
    "e2v": 1024,
    "mert": 1024,
    "aves2": 768,
    "clap": 512,
}