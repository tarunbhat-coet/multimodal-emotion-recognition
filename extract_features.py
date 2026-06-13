import os
import numpy as np
import cv2

# ================= CONFIG =================
DATA_ROOT = "/content/drive/MyDrive/Colab Notebooks/emotion_recognition/data/iemocap"

emotion_map = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3
}

MAX_FRAMES = 30
FRAME_SIZE = 48


# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ================= VIDEO PROCESSING =================
def extract_video_frames(video_path):
    if video_path is None or not os.path.exists(video_path):
        return np.zeros((MAX_FRAMES, FRAME_SIZE * FRAME_SIZE), dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return np.zeros((MAX_FRAMES, FRAME_SIZE * FRAME_SIZE), dtype=np.float32)

    indices = np.linspace(0, total - 1, MAX_FRAMES, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            frames.append(np.zeros(FRAME_SIZE * FRAME_SIZE))
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
        else:
            face = gray

        face = cv2.resize(face, (FRAME_SIZE, FRAME_SIZE))
        face = face.astype(np.float32) / 255.0

        frames.append(face.flatten())

    cap.release()
    return np.array(frames, dtype=np.float32)


# ================= HELPERS =================
def load_transcripts(file):
    d = {}
    with open(file, 'r') as f:
        for line in f:
            if ":" not in line:
                continue

            k, v = line.split(":", 1)
            k = k.strip().split()[0]  # remove timestamp
            d[k] = v.strip()

    return d


def load_labels(file):
    d = {}
    with open(file, 'r') as f:
        for line in f:
            if "[" not in line:
                continue

            parts = line.strip().split()

            utt_id = None
            emo = None

            for p in parts:
                if p.startswith("Ses"):
                    utt_id = p
                if p in emotion_map:
                    emo = p

            if utt_id and emo:
                d[utt_id] = emotion_map[emo]

    return d


# ================= MAIN =================
def extract_all():

    audio_paths = []
    video_features = []
    text_all = []
    labels = []

    print("🔍 Scanning dataset...")

    for session in os.listdir(DATA_ROOT):
        if not session.startswith("Session"):
            continue

        print(f"Processing {session}...")

        sp = os.path.join(DATA_ROOT, session)

        wav_root = os.path.join(sp, "sentences/wav")
        tr_root  = os.path.join(sp, "dialog/transcriptions")
        lb_root  = os.path.join(sp, "dialog/EmoEvaluation")
        vid_root = os.path.join(sp, "dialog/dialogueVideos")

        for dialog in os.listdir(wav_root):

            wav_dir = os.path.join(wav_root, dialog)
            tr_file = os.path.join(tr_root, dialog + ".txt")
            lb_file = os.path.join(lb_root, dialog + ".txt")

            if not os.path.exists(tr_file) or not os.path.exists(lb_file):
                continue

            transcripts = load_transcripts(tr_file)
            label_dict = load_labels(lb_file)

            for f in os.listdir(wav_dir):

                if not f.endswith(".wav"):
                    continue

                utt = f.replace(".wav", "")

                if utt not in transcripts or utt not in label_dict:
                    continue

                wav_path = os.path.join(wav_dir, f)

                # ===== VIDEO =====
                video_path = os.path.join(vid_root, dialog + ".avi")
                if not os.path.exists(video_path):
                    video_path = None

                video_feat = extract_video_frames(video_path)

                # ===== SAVE =====
                audio_paths.append(wav_path)
                video_features.append(video_feat)
                text_all.append(transcripts[utt])
                labels.append(label_dict[utt])

    if len(text_all) == 0:
        print("\n❌ ERROR: No samples extracted!")
        return

    print(f"\n✅ Total samples collected: {len(text_all)}")

    print("💾 Saving features...")

    os.makedirs("features", exist_ok=True)

    np.save("features/audio_raw.npy", np.array(audio_paths, dtype=object))
    np.save("features/video.npy", np.array(video_features, dtype=np.float32))
    np.save("features/text_raw.npy", np.array(text_all, dtype=object))
    np.save("features/labels.npy", np.array(labels))

    print("✅ DONE")
    print("Audio:", len(audio_paths))
    print("Video:", np.array(video_features).shape)
    print("Text :", len(text_all))
    print("Labels:", len(labels))


if __name__ == "__main__":
    extract_all()