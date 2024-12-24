# Dog Emotion Detection with YOLOv5 and MobileNetV3

This repository contains the implementation of a real-time dog emotion detection system using YOLOv5 for dog face detection and MobileNetV3 for emotion classification. The project aims to enhance human-dog interaction by accurately identifying canine emotions such as Happy, Sad, Relaxed, and Angry.

## Features
- Real-time dog face detection using YOLOv5
- Emotion classification into four categories: Happy, Sad, Relaxed, Angry
- Pre-trained YOLOv5 and MobileNetV3 models for ease of use
- Custom datasets for dog face detection and emotion recognition

---

## How to Run the Code
To run the code, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/atticus453/Intro-2-ML-Final.git
    cd Intro-2-ML-Final
    ```

2. Run the script `a.py` with the following command:
    ```bash
    python a.py best.pt output_folder --conf 0.8
    ```
    - Replace `best.pt` with the path to the YOLOv5 model weights.
    - Replace `output_folder` with the desired directory to store output results.
    - Adjust `--conf` to set the confidence threshold (default: `0.8`).

---

## Datasets
1. **Dog Face Detection Dataset**
   - 6,200 images:
     - 300 manually labeled with LabelImg
     - 5,900 sourced from the [Dog Face Detection Dataset (YOLO Format)](https://www.kaggle.com/datasets/wutheringwang/dog-face-detectionyolo-format)

2. **Dog Emotion Recognition Dataset**
   - 4,000 images (1,000 per emotion category: Happy, Sad, Relaxed, Angry):
     - 800 training images
     - 200 validation images
   - Sourced from the [Dog Emotion Dataset](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion)

---

## Software Requirements
- Python 3.10
- PyTorch 1.13.0
- CUDA 12.4 (for GPU acceleration)
- Required libraries: `numpy`, `pandas`, `seaborn`, `opencv-python`, `ultralytics`

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Results
- Overall Emotion Classification Accuracy: **68%**
- Class-wise Accuracy:
  - Happy: 75%
  - Sad: 72%
  - Relaxed: 65%
  - Angry: 60%

---

## Authors
- **Chia Yu, Lin**: Analyze Dog emotion detection and model-related tasks.
- **Kai Chi, Hsu**: Analyze dog face detection methods.
- **Po Shen, Huang**: Analyze dog face detection methods.
- **Jin Wei, Chang**: Analyze dog face detection methods.
- **I Hsuan, Chu**: Analyze Dog emotion detection and model-related tasks.

---

## Data and Code Availability
The code and pre-trained models are available in this repository. The datasets are included but due to size constraints, some files may need to be downloaded directly from Kaggle (links provided above).

---

Feel free to contribute, raise issues, or share feedback to improve this project!
