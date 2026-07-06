# 😊 Emotion Recognition System

A real-time emotion detection system that recognizes human emotions from facial expressions using Deep Learning and Computer Vision.

---

## 🎯 What It Does

- Detects faces using your webcam
- Recognizes 7 emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
- Analyzes uploaded images
- Shows emotion trends over time with a graph
- Customizable themes (Blue, Pink, Yellow, Gray)

---

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow / Keras (Deep Learning)
- OpenCV (Face Detection)
- Tkinter (GUI)
- Matplotlib (Graphs)
- FER2013 Dataset

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/WintNadi/emotion_r.git
cd emotion_r
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python emotions.py
```

---

## 📸 Features in Action

### Main Menu
- Click **"Real-time Detection"** to start webcam
- Click **"Upload Image"** to analyze a photo
- Click **"Theme Settings"** to change colors

### Real-Time Detection
- Shows emotion label with confidence percentage
- Displays emotion timeline graph
- Tracks emotions over time

### Image Upload
- Upload any image with a face
- Detects and labels all faces
- Download the annotated image

---

## 🧠 How It Works

1. **Face Detection** – OpenCV's Haar Cascade finds faces
2. **Emotion Classification** – CNN model predicts emotions
3. **Logic Mapping** – pyDatalog converts numbers to emotion names
4. **Visualization** – Matplotlib shows emotion trends

---

## 📊 Model Details

| Item | Details |
|------|---------|
| **Dataset** | FER2013 (35,887 images) |
| **Model** | Convolutional Neural Network (CNN) |
| **Input Size** | 48×48 grayscale |
| **Emotions** | 7 (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised) |
| **Training** | Google Colab with GPU |

---

## ⚠️ Known Limitations

- Works best in good lighting
- Struggles with side profiles or tilted faces
- May not detect faces with masks or glasses
- Background clutter can cause false positives

---

## 🔮 Future Improvements

- Better accuracy with advanced models
- Mobile app version
- Real-time alerts
- Mood-based recommendations (music, activities)

---

## 📚 References

- [FER2013 Dataset](https://www.kaggle.com/datasets/deadskull7/fer2013)
- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

## 🙏 Acknowledgments

- FER2013 dataset providers
- OpenCV community
- Google Colab for free GPU

---

**Built with ❤️ by Team Emotion Recognition**
