# Real-Time Sign Language Detection using MediaPipe and LSTM

## Overview

This project builds a real-time sign language recognition system using computer vision and deep learning.

MediaPipe is used to extract hand and body landmarks from video frames. These landmarks are converted into sequential data and fed into an LSTM neural network that learns gesture patterns over time.

The trained model predicts sign language gestures from a webcam feed and displays the detected word in real time.

The project is divided into two stages.

1. Data collection of gesture sequences  
2. Model training and real-time inference

---

## Project Structure

```
project-folder
│
├── 01_data_collection_main.ipynb
├── 02_train_and inference_main.ipynb
│
├── MP_Data/
│   ├── gesture_1/
│   ├── gesture_2/
│   └── ...
│
├── actions.npy
├── sign_model.keras
│
└── README.md
```

---

## Technologies Used

Python  
OpenCV  
MediaPipe  
TensorFlow / Keras  
NumPy  
Matplotlib  

---

## System Pipeline

### 1. Landmark Extraction

MediaPipe Holistic detects body and hand landmarks from each frame captured by the webcam.

The system extracts the following features

Pose landmarks  
Left hand landmarks  
Right hand landmarks  

Each frame produces a feature vector containing the coordinates of all detected landmarks.

Feature distribution

Pose landmarks → 132 values  
Left hand landmarks → 63 values  
Right hand landmarks → 63 values  

Total features per frame

258 features

---

### 2. Sequence Creation

Instead of classifying a single frame, the model learns temporal motion patterns.

Each gesture is recorded as a sequence.

Sequence length

40 frames per sequence

Example structure

```
MP_Data
 ├ Hello
 │   ├ sequence_0
 │   ├ sequence_1
 │   └ sequence_2
 ├ Thanks
 └ Yes
```

Each frame in the sequence is stored as a NumPy array.

---

## Part 1 Data Collection

Notebook

```
01_data_collection_main.ipynb
```

Purpose

Collect training data for each gesture using a webcam.

Main steps

Open webcam using OpenCV  
Detect landmarks using MediaPipe  
Extract landmark coordinates  
Save frame data as NumPy arrays  
Store sequences inside dataset folders  

Example configuration

```python
WORD = "Hello"
DATA_PATH = "MP_Data"
NO_SEQUENCES = 60
SEQUENCE_LENGTH = 40
```

This records

60 sequences  
40 frames per sequence

---

## Part 2 Model Training and Inference

Notebook

```
02_train_and inference_main.ipynb
```

This notebook performs

Dataset loading  
Model training  
Real-time gesture prediction

---

## Dataset Preparation

Saved sequences are loaded and converted into training arrays.

Input shape

```
(number_of_sequences, 40, 258)
```

Labels are encoded using categorical encoding.

---

## LSTM Model Architecture

The model learns temporal patterns from gesture sequences.

Example structure

```
LSTM
LSTM
Dense
Dense
Softmax
```

The final softmax layer outputs the probability of each gesture class.

---

## Model Training

The model is trained using categorical cross-entropy loss and the Adam optimizer.

Training outputs include

Training accuracy  
Validation accuracy  
Loss curves

After training the model is saved as

```
sign_model.keras
```

---

## Real-Time Gesture Detection

The inference system runs using a webcam feed.

Pipeline

1 Capture video frame  
2 Extract MediaPipe landmarks  
3 Store frames in sequence buffer  
4 Pass sequence to trained LSTM model  
5 Predict gesture class  
6 Display predicted word on screen  

Prediction updates continuously as frames are captured.

Example output

```
Prediction: Hello
Confidence: 0.92
```

---

## Installation

Clone the repository

```
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection
```

Install required libraries

```
pip install opencv-python mediapipe tensorflow numpy matplotlib
```

---

## Running the Project

Step 1

Run the data collection notebook

```
01_data_collection_main.ipynb
```

Record gesture sequences.

Step 2

Train the model

```
02_train_and inference_main.ipynb
```

Step 3

Run the inference section to perform real-time sign detection.

---

## Model Files

actions.npy  
Stores gesture labels used during training.

sign_model.keras  
Saved trained LSTM model used for inference.

---

## Future Improvements

Increase number of gestures  
Improve dataset size  
Add sentence level translation  
Deploy the model using Streamlit or a web interface  
Improve prediction confidence filtering

---

## Applications

Assistive communication systems  
Accessibility tools  
Human computer interaction  
Gesture-based interfaces
