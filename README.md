# 🚗 Driver Behaviour Detection Using YOLO

A real-time driver behaviour detection system built using a custom-trained YOLO model. The project detects driver eye conditions, phone usage, and yawning to demonstrate potential driver drowsiness and distraction monitoring.

## 🚀 Live Demo

**[▶ Launch Driver Behaviour Detection Demo](https://arpitaai.github.io/-Driver-Behaviour-Detection/)**

Try the browser-based demo using your webcam. Allow camera access when prompted.

## 📌 Project Overview

This project implements real-time driver behaviour monitoring using a custom-trained YOLO model.

The system identifies different driver behaviours through visual detection and displays predictions directly in the browser.

## 🎯 Detection Classes

The model detects six driver behaviour categories:

- Eye Close
- Eye Close with Glasses
- Eye Open
- Eye Open with Glasses
- Phone Use
- Yawn

## 🖥️ Live Application Features

- Browser-based webcam demonstration
- Custom YOLO model integration
- ONNX model inference
- Real-time camera interaction
- Visual detection results
- Start and stop camera controls

## 🧠 Technologies Used

- Python
- YOLO
- PyTorch
- ONNX
- OpenCV
- HTML
- CSS
- JavaScript

## 📂 Project Structure

```text
Driver-Behaviour-Detection/
│
├── desktop/
│   └── Original Python/OpenCV application
│
├── model/
│   └── best.onnx
│
├── index.html
├── style.css
├── app.js
├── requirements.txt
└── README.md
```

## ⚙️ Run the Local Python Version

Clone the repository:

```bash
git clone https://github.com/ArpitaAI/-Driver-Behaviour-Detection.git
```

Navigate to the desktop directory:

```bash
cd Driver-Behaviour-Detection/desktop
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r ../requirements.txt
```

Run the detection application:

```bash
python yolo_detect.py --model best.pt --source usb0
```

## 🌐 Deployment

The browser-based application is hosted using **GitHub Pages**.

The live demo uses the ONNX model for browser-based inference.

## 👩‍💻 Author

**Arpita Islam**

GitHub: [@ArpitaAI](https://github.com/ArpitaAI)

---

⭐ If you find this project interesting, feel free to explore the repository and try the live demo.



