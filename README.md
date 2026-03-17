# YOLOv8 Object Detection with Text-to-Speech 🎯

## 📌 Overview

This project performs **real-time object detection** on images using the **YOLOv8 deep learning model**. Detected objects are not only highlighted with bounding boxes but are also **converted to speech** using gTTS, making the project interactive and accessible.

## 🛠️ Technologies Used

* **Python 3.x**
* **Ultralytics YOLOv8** (lightweight YOLOv8 nano model)
* **OpenCV** (image processing)
* **Matplotlib** (visualization)
* **gTTS** (Google Text-to-Speech)
* **IPython.display** (play audio in Colab)
* **Google Colab** for notebook execution

---

## ⚙️ Features

* Upload any image for object detection
* Detect multiple objects using **YOLOv8 nano**
* Draw **bounding boxes and class labels** on detected objects
* Convert detected object names to **audio using gTTS**
* Display final image with detections in Colab


## 📊 How It Works

1. **Upload Image** – Use the Colab file uploader to provide an image.
2. **Load YOLOv8 Model** – Uses pre-trained YOLOv8n weights (`yolov8n.pt`) for fast inference.
3. **Detect Objects** – YOLOv8 performs detection and outputs bounding boxes and class labels.
4. **Visualize** – Draws bounding boxes and labels on the image using OpenCV.
5. **Text-to-Speech** – Converts the detected object names to audio using gTTS and plays it in Colab.

---

## ▶️ How to Run

1. Open this notebook in **Google Colab**
2. Run the first cell to install dependencies:

```bash
!pip install ultralytics gtts IPython
```

3. Upload an image using the file uploader widget
4. Run all cells sequentially to:

   * Detect objects
   * Display image with bounding boxes
   * Hear detected objects via speech


##  Results

* Fast detection on **YOLOv8n (nano)**: ~329 ms per image
* Supports **multiple objects per image**
* Interactive **audio feedback** using gTTS

---

