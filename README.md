# Computer Vision Project: Camera Calibration, Augmented Reality, Stereo Vision, Feature Matching
This project is a comprehensive PyQt5-based application for various computer vision tasks, including camera calibration, augmented reality (AR), stereo disparity map generation, and SIFT feature matching.

## Features

- **Camera Calibration**: Calibrate a camera using a chessboard pattern.
- **Augmented Reality**: Overlay 3D text onto a calibrated scene.
- **Stereo Disparity Map**: Generate a disparity map from stereo images.
- **SIFT Feature Matching**: Detect and match keypoints between two images.

## Prerequisites

Before running the application, ensure you have the following libraries installed:

```bash
pip install opencv-python numpy pyqt5
```

## Directory Structure

The project expects the following directory structure:

```
project_root/
│
├── ui.py                  # Generated UI file
├── main.py                         # Main application file
└── data/
    └── ...                         # Chessboard images
```

## Notes

- The application uses OpenCV's `findChessboardCorners` function to detect chessboard corners.
- For AR, the application loads 3D coordinates from `.txt` files.
- The stereo disparity map is generated using OpenCV's `StereoBM` algorithm.
- SIFT feature matching uses the Brute-Force matcher with a ratio test for robust matches.

## Acknowledgments

- OpenCV documentation for camera calibration and feature matching.
- PyQt5 for creating the graphical user interface.

---
