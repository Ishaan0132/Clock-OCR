# Clock Time Detection

A computer vision project that extracts precise time (HH:MM:SS) from images of analog clocks using two approaches: Classical Computer Vision and Deep Learning (CNN).

---

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Computer Vision Approach](#computer-vision-approach)
  - [CNN Approach](#cnn-approach)
- [How It Works](#how-it-works)
- [Contributors](#contributors)

---

## About the Project

This project implements two distinct pipelines for reading time from analog clock images:

### Computer Vision Approach
Uses traditional image processing techniques including:
- Image preprocessing (CLAHE, Thresholding)
- Hough Circle Transform for clock face detection
- Contour detection and Probabilistic Hough Transform for hand segmentation
- Vector geometry calculations (dot/cross product) for angle computation

### Deep Learning Approach
Treats the problem as image classification:
- Clock images categorized into 144 classes (12 hours x 12 five-minute intervals)
- Custom CNN architectures trained on a labeled dataset
- Two model variants: 38x38 and 64x64 input resolutions

---

## Features

- Detects time from analog clock images
- Identifies hour, minute, and second hands
- Handles various clock styles and designs
- Provides visualization of detected hands
- Supports both CV and CNN-based approaches

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Ip-Lab.git
cd Ip-Lab
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

- opencv-python
- numpy
- tensorflow (>= 2.0)
- matplotlib

---

## Usage

### Computer Vision Approach

1. Place your clock image in the project directory (e.g., `images/clock.jpg`)

2. Run the CV module:

```bash
python -m cv.main
```

3. Or use it programmatically:

```python
from cv.main import solve, main
import cv2

# Using the solve function
img = cv2.imread("path/to/clock.jpg")
result_img, time_str = solve(img)
print(f"Detected Time: {time_str}")

# Or using the main function with a path
main("path/to/clock.jpg")
```

**Output:**
- Console: Prints detected time (e.g., `Detected Time: 10:23:45`)
- Window: Displays the image with detected hands annotated

### CNN Approach

1. Prepare your dataset:
   - Organize images in `data/train/` and `data/test/` directories
   - Each class should be in its own subfolder (e.g., `data/train/10-00/`, `data/train/10-05/`)

2. Run training:

```bash
python -m cnn.train
```

3. Training configuration can be modified in `cnn/train.py`:
   - `epochs`: Number of training epochs (default: 50)
   - `batch_size`: Batch size for training (default: 32)
   - `data_dir`: Path to training data (default: `"data/train"`)

**Output:**
- Training progress and accuracy metrics
- Plots of training/validation accuracy
- Test set evaluation results

---

## How It Works

### CV Pipeline

1. **Preprocessing**: Image is resized and converted to HSV color space, then enhanced with CLAHE and binary thresholding

2. **Clock Detection**: Hough Circle Transform detects the clock face; falls back to contour detection if circles aren't found

3. **Hand Segmentation**: Probabilistic Hough Transform detects lines, which are grouped by angle and filtered by length/thickness to identify hour, minute, and second hands

4. **Time Calculation**: Angles relative to 12 o'clock position are computed using vector geometry and converted to time

### CNN Pipeline

1. **Data Preparation**: Images are loaded, resized (38x38 or 64x64), converted to grayscale, and augmented

2. **Model Architecture**: Two similar CNN architectures with:
   - Convolutional layers (32, 64, 128 filters)
   - Batch normalization and dropout for regularization
   - Dense output layer with 144 classes (softmax activation)

3. **Training**: Models trained with Adam optimizer and categorical cross-entropy loss

---

## Contributors

- [Ishaan Shaikh](https://github.com/Ishaan0132/)
- [Sourish Phate](https://github.com/sourishphate/)
- [Krish Shah](https://github.com/KrishShah3011/)
- [Kaustubh Sonawane](https://github.com/Kaustubh6077)