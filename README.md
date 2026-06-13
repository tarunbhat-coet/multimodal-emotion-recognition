# 🎭 Multimodal Emotion Recognition (IEMOCAP)

> A deep learning pipeline for speech emotion recognition using **1D CNN** on audio feature vectors extracted from the **IEMOCAP** dataset — with modular scripts for feature extraction, spectrogram generation, model training, and real-time inference.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-IEMOCAP-orange?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Emotions Detected](#emotions-detected)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Real-Time Demo](#real-time-demo)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

This project builds a **Speech Emotion Recognition (SER)** system using the IEMOCAP dataset. The pipeline is broken into small, focused modules — each script or notebook handles one task (label extraction, feature extraction, spectrogram generation, training, testing) — making the workflow easy to follow and extend.

**Core approach:**
- Extract audio feature vectors (MFCCs, spectrograms) from IEMOCAP sentence-level `.wav` files
- Train a **1D CNN** classifier on the extracted features
- Evaluate per-emotion accuracy and run real-time inference from a microphone

---

## 🎭 Emotions Detected

This project focuses on **4 emotion classes** from IEMOCAP:

| Label | Emotion | IEMOCAP Code |
|-------|---------|--------------| 
| 😠 | Angry | `ang` |
| 😢 | Sad | `sad` |
| 😊 | Happy | `hap` |
| 😐 | Neutral | `neu` |

---

## 📦 Dataset

### About IEMOCAP

The **Interactive Emotional Dyadic Motion Capture (IEMOCAP)** database is a multimodal, multispeaker acted emotion dataset collected at the SAIL Lab at USC.

| Property | Details |
|----------|---------|
| 📍 Source | [USC SAIL Lab](https://sail.usc.edu/iemocap/) |
| ⏱️ Duration | ~12 hours of audiovisual data |
| 🎬 Sessions | 5 dyadic sessions |
| 👥 Speakers | 10 actors (5 male, 5 female) |
| 📝 Modalities | Audio, Video, Motion Capture, Text |
| 🏷️ Labels | Categorical (anger, happiness, sadness, neutrality, ...) + Dimensional (valence, activation, dominance) |

> **Access:** IEMOCAP requires a formal request. Register at [this page](https://sail.usc.edu/iemocap/release_form.php) to get access.

---

### ⚠️ Important: Dialogue vs. Sentence Format

The IEMOCAP dataset provides `.wav` files in **two formats**:

| Format | Description | Issue |
|--------|-------------|-------|
| **Dialogue** | Full conversation per session | Produces a very large number of audio vectors — not feasible due to memory constraints |
| **Sentence** | Individual utterances ✅ | Manageable size, clean boundaries |

**Solution used in this project:**

All **sentence-level** `.wav` files were copied into a **single flat folder**, which simplifies file access across all scripts and avoids any path traversal logic:
---

### ⚡ Quick Start with Kaggle (Skip Raw Dataset Setup)

If you want to jump straight to model training without requesting IEMOCAP access, pre-extracted audio feature vectors are available on Kaggle:

👉 **[IEMOCAP Audio Vectors CSV — Kaggle](https://www.kaggle.com/aditya310794/iemocap-audio-vectors-csv/notebooks)**

Download the CSV and place it in the `data/` folder to skip Steps 1–3 in the usage guide below.

---

## 📁 Project Structure
> ✅ = **Recommended entry point.** Always prefer the `_new` versions — they contain the latest improvements.

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip
- A CUDA-capable GPU is recommended for training (CPU works but is slower)

### 1. Clone the Repository

```bash
git clone https://github.com/tarunbhat-coet/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option A — Full Pipeline (Raw IEMOCAP)

**Step 1: Prepare the data**

Copy all sentence-level `.wav` files from IEMOCAP into a single folder:
```bash
# Example structure after copying
data/wav_sentences/*.wav
```

**Step 2: Extract emotion labels**
```bash
jupyter notebook 1_extract_emotion_labels.ipynb
```

**Step 3: Build audio feature vectors**
```bash
jupyter notebook 2_build_audio_vectors.ipynb
# or
python extract_features_new.py
```

**Step 4: Generate spectrograms** *(optional — for CNN on images)*
```bash
python 3_generate_spectrogram_1.py
python 4_generate_spectrogram_2.py
python 5_generate_spectrogram_3.py
```

**Step 5: Train**
```bash
python train_new.py
# or to compare all model variants:
python train_all_models.py
```

**Step 6: Evaluate**
```bash
python test_new.py
```

---

### Option B — Quick Start (Kaggle CSV)

```bash
# 1. Download CSV from Kaggle (link above) → place in data/
# 2. Skip directly to training
python train_new.py
```

---

### Real-Time Inference

```bash
python realtime_application_new.py
```

Captures live microphone audio, extracts features in real time, and displays the predicted emotion with confidence score.

---

## 🏗️ Model Architecture

The core model is a **1D CNN** operating on audio feature vectors (MFCCs):
---

## 📊 Results

> Results on IEMOCAP (4-class: Angry, Sad, Happy, Neutral):

| Modality | Description | Accuracy |
|----------|-------------|----------|
| t | Text only | 78.40% |
| a | Audio only | 78.95% |
| v | Video only | 78.95% |
| ta | Text + Audio | 78.84% |
| tv | Text + Video | 78.62% |
| av | Audio + Video | 78.51% |
| **tav** | **Text + Audio + Video (Full Fusion)** | **79.51%** ✅ |

> 🏆 Best model: **Full Multimodal Fusion (tav) at 79.51%** — achieved with early stopping at epoch 16.

## 🎤 Real-Time Demo

```bash
python realtime_application_new.py
```

**How it works:**
1. Captures audio in short sliding windows (~2–3 seconds) from your microphone
2. Extracts MFCC features from each window
3. Runs inference through the trained 1D CNN
4. Prints predicted emotion + confidence to the terminal (or overlays on webcam feed)

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome!

1. Fork the repository
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tarun Bhat**
- 🐙 GitHub: [@tarunbhat-coet](https://github.com/tarunbhat-coet)

---

> ⭐ If this project helped you, please consider giving it a star!
