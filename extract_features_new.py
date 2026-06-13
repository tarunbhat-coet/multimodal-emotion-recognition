import numpy as np
import torch
import librosa
from transformers import BertTokenizer, BertModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

# ================= LOAD DATA =================
text = np.load("features/text_raw.npy", allow_pickle=True)
audio = np.load("features/audio_raw.npy", allow_pickle=True)
video = np.load("features/video.npy")
labels = np.load("features/labels.npy")

# ================= LOAD MODEL =================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert.eval()

# ================= STORAGE =================
text_feats = []
audio_feats = []
video_feats = []

print("Extracting features...")

for i in range(len(text)):
    try:
        # ===== TEXT (BERT MEAN POOLING) =====
        enc = tokenizer([text[i]], return_tensors="pt", padding=True, truncation=True)
        input_ids = enc['input_ids'].to(DEVICE)
        attn_mask = enc['attention_mask'].to(DEVICE)

        with torch.no_grad():
            out = bert(input_ids, attention_mask=attn_mask).last_hidden_state
            t = torch.cat([
    out[:, 0, :],          # CLS
    out.mean(dim=1)        # mean
], dim=1)

        text_feats.append(t.cpu().numpy()[0])

        # ===== AUDIO (MFCC - FINAL FIX) =====
        waveform, sr = librosa.load(audio[i], sr=16000)

        # trim silence
        waveform, _ = librosa.effects.trim(waveform)

        # normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform))

        # MFCC (emotion-friendly)
        mfcc = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=40)
        mfcc = np.mean(mfcc, axis=1)

        audio_feats.append(mfcc)

        # ===== VIDEO =====
        v = video[i]
        v = v[::4]
        v = np.mean(v, axis=0)

        video_feats.append(v)

    except Exception as e:
        print(f"Skipping {i}: {e}")
        continue

    if i % 200 == 0:
        print(f"Processed: {i}/{len(text)}")

# ================= SAVE =================
np.save("features/text_bert.npy", np.array(text_feats))
np.save("features/audio_wav2vec.npy", np.array(audio_feats))  # keep same name
np.save("features/video_processed.npy", np.array(video_feats))
np.save("features/labels.npy", labels)

print("✅ DONE")