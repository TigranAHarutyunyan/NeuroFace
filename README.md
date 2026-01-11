# 💡 Project Title: Neuro Face

## 🧠 Short Description

**NeuroFace** is a Python-based video processing tool designed to detect, extract, and enhance human faces in videos using deep learning. It combines object detection (YOLOv8), clustering, and enhancement models (GFPGAN) to deliver clean, high-quality face crops for further use in analysis or media generation.

---

## 🎯 Objectives

-   Automate face detection and extraction from videos.
-   Enhance face resolution using GFPGAN.
-   Cluster similar faces for identity grouping.
-   Track frame quality and choose the best frames.
-   Optimize for GPU-based batch processing.

---

## 🔬 Methodology

-   Use **YOLOv8** for real-time face detection.
-   Apply sharpness & brightness filters to select top frames.
-   Enhance facial crops using **GFPGAN 1.2-clean**.
-   Cluster detected faces using **HDBSCAN** based on embeddings.
-   Store outputs by identity with clean folder structure.

---

## ⚙️ Technologies Used

-   **Language:** Python 3.10
-   **Libraries:** OpenCV, PyTorch, NumPy, HDBSCAN, GFPGAN, Ultralytics
-   **Models:**
    -   YOLOv8-face for detection
    -   GFPGAN for restoration
    -   (Optional) ESRGAN for further enhancement
-   **Environment:** CUDA 12.1, RunPod-compatible

---

## 🚧 Challenges and Solutions

| Challenge                | Solution                                                         |
| ------------------------ | ---------------------------------------------------------------- |
| Low-res or blurry frames | Filter with sharpness metric; restore with GFPGAN                |
| Overlapping detections   | Confidence threshold tuning in YOLOv8                            |
| Performance bottlenecks  | Use GPU with batched DataLoader, cache clearing, multiprocessing |
| Misclustered identities  | Switched to HDBSCAN over DBSCAN; fine-tuned min_cluster_size     |

---

## 📈 Results

-   ✅ 85–95% detection accuracy on multiple videos
-   🔍 3x–4x image clarity improvement on blurry faces
-   📂 Face crops saved by frame and cluster ID
-   🧠 Smart filtering ensures best-quality enhancement input

---

## 🔮 Future Work

-   Add webcam / RTSP support
-   Train custom YOLO & GFPGAN models
-   Export results as searchable timeline (timestamped detections)
-   Add REST API for integration in larger pipelines
-   Build Streamlit/Gradio UI for non-technical users

---

## 🔗 References

1. [OpenCV Documentation](https://docs.opencv.org)
2. [GFPGAN 1.2-clean GitHub](https://github.com/TencentARC/GFPGAN)
3. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
4. [ESRGAN: Super Resolution](https://arxiv.org/abs/1809.00219)

---

## ⚡ Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the program
python3 Neuro_Face.py
```

👉 Pretrained models available at:

-   YOLOv8 + weights: [Download](https://drive.google.com/drive/folders/1tJ-IfF_luVGGiTgi2PMjpRW05dRvi8_-?usp=drive_link)
-   GFPGAN weights: [Download](https://drive.google.com/drive/folders/1gszj4kZUUVepviidgykzW4Yfb4jM-zek?usp=drive_link)

---
