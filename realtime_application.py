import cv2
import torch
import numpy as np
import sounddevice as sd
import librosa

from models.text_encoder import TextEncoder, encode_text
from models.audio_encoder import AudioEncoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'best_model.pt'

EMOTIONS = ["Angry", "Happy", "Sad", "Neutral"]

FRAME_SIZE = 48
MAX_FRAMES = 30


# ================= VIDEO =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_video_frames(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
    else:
        face = gray

    face = cv2.resize(face, (FRAME_SIZE, FRAME_SIZE))
    face = face.astype(np.float32) / 255.0

    flat = face.flatten()
    return np.tile(flat, (MAX_FRAMES, 1))  # [30,2304]


# ================= AUDIO =================
def record_audio(duration=3, sr=16000):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()


# ================= MODEL =================
class MultimodalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.text_enc = TextEncoder()
        self.audio_enc = AudioEncoder()

        self.video_fc = torch.nn.Linear(48*48, 128)

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 4)
        )

    def forward(self, texts, audio_wave, video):

        # TEXT
        input_ids, attention_mask = encode_text(texts)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        t = self.text_enc(input_ids, attention_mask)

        # AUDIO
        a = self.audio_enc(audio_wave)

        # VIDEO
        v = video.to(DEVICE)
        v = self.video_fc(v)
        v = v.mean(dim=1)

        # ATTENTION
        x = torch.stack([t, a, v], dim=1)
        attn_out, _ = self.attn(x, x, x)
        fused = attn_out.mean(dim=1)

        return self.fc(fused)


# ================= LOAD MODEL =================
print("Loading model...")
model = MultimodalModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ================= WEBCAM =================
cap = cv2.VideoCapture(0)

print("\n🎥 Press 'e' to detect emotion | 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Emotion Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('e'):
        print("\n🔍 Detecting emotion...")

        # ===== TEXT =====
        text_input = input("Enter sentence: ")
        texts = [text_input]

        # ===== AUDIO =====
        audio = record_audio()
        audio_wave = [audio]  # list for wav2vec

        # ===== VIDEO =====
        video_feat = extract_video_frames(frame)
        video_tensor = torch.tensor(video_feat, dtype=torch.float32).unsqueeze(0)

        # ===== PREDICTION =====
        with torch.no_grad():
            output = model(texts, audio_wave, video_tensor)
            pred = torch.argmax(output, dim=1).item()

        print(f"🎯 Emotion: {EMOTIONS[pred]}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()