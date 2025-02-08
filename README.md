# Chem-is-Try

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Framework](https://img.shields.io/badge/Framework-MediaPipe_|_YOLO-orange.svg)

> Intelligent Chemical Experiment Operation Detection System Based on Computer Vision

## 📖 Project Introduction

Chem-is-Try is an intelligent chemical experiment assistance system based on Orange Pi (RK3588), implementing real-time monitoring and recognition of experimental operations through computer vision technology. The system can accurately identify laboratory equipment and detect operational actions, providing intelligent assistance for chemical experiments.

### ✨ Core Features

- 🔍 Laboratory Equipment Instance Segmentation (YOLO11-seg)
- 👋 Precise Hand Action Recognition (MediaPipe)
- ⚡ Multi-threaded Parallel Inference Architecture
- 🎯 Support for Various Operation Detection:
  - Equipment Grasping Recognition
  - Liquid Pouring Detection
  - Titration Operation Monitoring
  - More operations under development...

## 🚀 Quick Start

### Requirements

- Orange Pi or other RK3588 development board
- Python 3.8+
- USB Camera

### Installation

1. Clone the project
```bash
git clone https://github.com/yourusername/Chem-is-Try.git
cd Chem-is-Try
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

### Usage Examples

```bash
# Run gesture recognition demo
python static_gesture.py

# Run grasp detection demo
python grab_something.py

# Run instance segmentation demo
python Instance_Segmentation/yolo/yolo_seg.py
```

## 📁 Project Structure

```
Chem-is-Try/
├── datasets/              # Dataset resources
├── models/               # Pre-trained models
├── playground/           # Work-in-progress code
├── process_data/         # Data processing tools
├── utils/               # Utility functions
├── voices/              # Voice prompt resources
├── refrences.old        # Old references(Not used)
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation

```

## 🛠️ Technical Implementation

### Equipment Detection
- Uses YOLO11-seg for precise instance segmentation
- Supports simultaneous recognition of multiple equipment types
- Real-time tracking of equipment position and status

### Gesture Recognition
- Extracts 21 hand landmarks using MediaPipe
- Real-time hand gesture state analysis
- Precise action trajectory tracking

### Performance Optimization
- Uses tflite models for accelerated inference

## 💡 Custom Training

### Data Preparation
```bash
# Convert dataset format
python process_data/coco2yolo.py
```

### Model Training
```bash
# Start training
python Object_Detection/yolo/yolo_train.py
```

## 🤝 Contributing

1. Fork this project
2. Create a new feature branch
3. Submit your changes
4. Create a Pull Request

## 📄 License

This project is licensed under the [MIT](LICENSE) License.

## 🙏 Acknowledgments

- [YOLO](https://github.com/ultralytics/yolov5)
- [MediaPipe](https://mediapipe.dev/)

## 📮 Contact

- Project Lead: [Your Name]
- Email: your.email@example.com
- Issues: [GitHub Issues](https://github.com/yourusername/Chem-is-Try/issues)