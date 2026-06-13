import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

from models.text_encoder import TextEncoder, encode_text
from models.audio_encoder import AudioEncoder, load_audio

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_model.pt'
BATCH_SIZE = 4

print("Device:", DEVICE)


# ================= DATASET =================
class IEMOCAPDataset(Dataset):
    def __init__(self):
        self.text = np.load("features/text_raw.npy", allow_pickle=True)
        self.audio = np.load("features/audio_raw.npy", allow_pickle=True)
        self.video = np.load("features/video.npy")
        self.labels = np.load("features/labels.npy")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.text[i], self.audio[i], self.video[i], self.labels[i]


ds = IEMOCAPDataset()
loader = DataLoader(ds, batch_size=BATCH_SIZE)


# ================= MODEL =================
class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_enc = TextEncoder()
        self.audio_enc = AudioEncoder()

        self.video_fc = nn.Linear(48*48, 128)

        self.attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)
        )

    def forward(self, texts, audios, videos):

        # ===== TEXT =====
        input_ids, attention_mask = encode_text(list(texts))
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        t = self.text_enc(input_ids, attention_mask)

        # ===== AUDIO =====
        audio_wave = [load_audio(p) for p in audios]
        a = self.audio_enc(audio_wave)

        # ===== VIDEO =====
        v = videos.to(DEVICE)          # [B,30,2304]
        v = self.video_fc(v)           # [B,30,128]
        v = v.mean(dim=1)              # [B,128]

        # ===== ATTENTION =====
        x = torch.stack([t, a, v], dim=1)  # [B,3,128]
        attn_out, _ = self.attn(x, x, x)
        fused = attn_out.mean(dim=1)

        return self.fc(fused)


# ================= LOAD MODEL =================
model = MultimodalModel().to(DEVICE)

print("Loading model...")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ================= TEST =================
preds, labels = [], []

with torch.no_grad():
    for texts, audios, videos, y in loader:

        outputs = model(texts, audios, videos)
        p = outputs.argmax(1)

        preds.extend(p.cpu().numpy())
        labels.extend(y.numpy())


print("\n📊 Classification Report:\n")
print(classification_report(labels, preds))