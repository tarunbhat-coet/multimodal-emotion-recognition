import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4

print("Device:", DEVICE)


# ================= NORMALIZATION =================
def norm(x):
    x = np.nan_to_num(x)
    mean = x.mean(axis=(0,1), keepdims=True)
    std = x.std(axis=(0,1), keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


# ================= DATASET =================
class IEMOCAPDataset(Dataset):
    def __init__(self):
        self.text = np.load('features/text/text.npy')
        self.audio = np.load('features/audio/audio.npy')
        self.labels = np.load('features/labels/labels.npy')

        self.text = self.text.astype(np.int64)
        self.audio = norm(self.audio)

        print("Dataset distribution:", np.bincount(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            torch.tensor(self.text[i], dtype=torch.long),
            torch.tensor(self.audio[i], dtype=torch.float32),
            torch.tensor(self.labels[i], dtype=torch.long)
        )


dataset = IEMOCAPDataset()

# ================= TRAIN/VAL SPLIT =================
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

labels = dataset.labels
class_counts = np.bincount(labels)

weights = len(labels) / (len(class_counts) * class_counts)
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

sample_weights = [1.0 / class_counts[l] for l in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


TEXT_VOCAB_SIZE = 5000
AUDIO_DIM = dataset.audio.shape[-1]
NUM_CLASSES = len(class_counts)


# ================= MODEL =================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # TEXT ENCODER
        self.text_embed = nn.Embedding(TEXT_VOCAB_SIZE, 128, padding_idx=0)
        self.text_lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.text_fc = nn.Linear(256, 128)

        # AUDIO ENCODER
        self.audio_lstm = nn.LSTM(AUDIO_DIM, 128, batch_first=True, bidirectional=True)
        self.audio_fc = nn.Linear(256, 128)

        # FUSION
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, text, audio):

        # TEXT
        t = self.text_embed(text)
        t, _ = self.text_lstm(t)
        t = self.text_fc(t.mean(dim=1))

        # AUDIO
        a, _ = self.audio_lstm(audio)
        a = self.audio_fc(a.mean(dim=1))

        # FUSION
        x = torch.cat([t, a], dim=1)
        return self.fc(x)


model = Model().to(DEVICE)

loss_fn = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.Adam(model.parameters(), lr=LR)

best = 0
patience = 10
no_improve = 0


# ================= TRAIN =================
for epoch in range(EPOCHS):

    model.train()
    correct = total = 0

    for text, audio, y in train_loader:
        text, audio, y = text.to(DEVICE), audio.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()
        out = model(text, audio)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total * 100

    # ===== VALID =====
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for text, audio, y in val_loader:
            text, audio, y = text.to(DEVICE), audio.to(DEVICE), y.to(DEVICE)

            out = model(text, audio)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

    val_acc = correct / total * 100

    print(f"Epoch {epoch+1}: Train {train_acc:.1f}% | Val {val_acc:.1f}%")

    if val_acc > best:
        best = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        no_improve = 0
        print("✅ Best model saved")
    else:
        no_improve += 1
        print(f"No improvement: {no_improve}/{patience}")

    if no_improve >= patience:
        print("\n🛑 Early stopping triggered")
        break

print("🔥 Best Val Acc:", best)