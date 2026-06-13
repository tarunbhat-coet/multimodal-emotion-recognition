import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4


# ================= NORMALIZATION =================
def normalize(x):
    x = np.nan_to_num(x)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x - mean) / std


# ================= DATASET =================
class DatasetIEMOCAP(Dataset):
    def __init__(self):
        self.t = normalize(np.load("features/text_bert.npy"))   # shape: 1536
        self.a = normalize(np.load("features/audio_wav2vec.npy"))  # MFCC (40)
        self.v = normalize(np.load("features/video_processed.npy"))
        self.y = np.load("features/labels.npy")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.t[i], dtype=torch.float32),
            torch.tensor(self.a[i], dtype=torch.float32),
            torch.tensor(self.v[i], dtype=torch.float32),
            torch.tensor(self.y[i], dtype=torch.long)
        )


dataset = DatasetIEMOCAP()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# ================= MODEL =================
class Model(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

        # FIXED DIMENSIONS
        self.text_fc = nn.Linear(1536, 768)   # 🔥 FIX
        self.audio_fc = nn.Linear(40, 768)
        self.video_fc = nn.Linear(2304, 768)

        # modality weights
        self.weights = nn.Parameter(torch.ones(3))

        self.attn = nn.MultiheadAttention(768, 4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

    def forward(self, t, a, v):

        feats = []
        weights = torch.softmax(self.weights, dim=0)

        idx = 0

        if "t" in self.mode:
            t = self.text_fc(t)   # 🔥 FIX HERE
            feats.append(t * weights[idx])
            idx += 1

        if "a" in self.mode:
            a = self.audio_fc(a)
            feats.append(a * weights[idx])
            idx += 1

        if "v" in self.mode:
            v = self.video_fc(v)
            feats.append(v * weights[idx])
            idx += 1

        x = torch.stack(feats, dim=1)

        attn_out, _ = self.attn(x, x, x)
        x = attn_out.mean(dim=1)

        return self.fc(x)


# ================= TRAIN FUNCTION =================
def train_model(mode):

    print(f"\n🔥 Training {mode}")

    model = Model(mode).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best = 0
    patience = 5
    no_imp = 0

    for epoch in range(EPOCHS):

        model.train()
        correct = total = 0

        for t, a, v, y in train_loader:
            t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(t, a, v)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total * 100

        # VALID
        model.eval()
        correct = total = 0

        with torch.no_grad():
            for t, a, v, y in val_loader:
                t, a, v, y = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), y.to(DEVICE)

                out = model(t, a, v)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total * 100

        print(f"Epoch {epoch+1}: Train {train_acc:.1f}% | Val {val_acc:.1f}%")

        # EARLY STOPPING
        if val_acc > best:
            best = val_acc
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= patience:
            print("🛑 Early stopping")
            break

    print(f"✅ Best ({mode}): {best:.2f}%")
    return best


# ================= RUN =================
modes = ["t", "a", "v", "ta", "tv", "av", "tav"]

results = {}

for m in modes:
    results[m] = train_model(m)

print("\n🔥 FINAL RESULTS")
for k, v in results.items():
    print(k, "→", v)