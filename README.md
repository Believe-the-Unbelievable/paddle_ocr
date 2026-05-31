## Vehicle Number Plate Detection using OCR and YOLO
A Streamlit web app that detects vehicle number plates using a trained YOLO model and recognizes the plate text using pre-trained PaddleOCR.

<img width="1708" height="902" alt="output" src="https://github.com/user-attachments/assets/2fefe9d4-fc74-4a8d-a273-e11b3acb51f5" />
<img width="1907" height="713" alt="output3" src="https://github.com/user-attachments/assets/c9856b46-fa83-4e64-9623-7102ceb35750" />
<img width="1758" height="645" alt="output2" src="https://github.com/user-attachments/assets/12e9182a-d371-4d39-acdf-3724cba1fe62" />

## Overview

This project combines number plate detection and OCR in one simple workflow. The YOLO model detects the number plate location from an uploaded vehicle image. The detected plate region is then cropped, resized, preprocessed, and passed to PaddleOCR for text recognition.

PaddleOCR is used as a pre-trained OCR model in this project. It was not trained or fine-tuned.

## Dataset

The YOLO model was trained using this Ultralytics dataset:

[Number Plate Dataset](https://platform.ultralytics.com/muhammed-azeem/datasets/number-plate)

The trained YOLO model file used in the app is:

```text
ocr.pt
```

## Workflow

1. Upload a vehicle image.
2. YOLO detects the number plate region.
3. The detected plate is cropped.
4. The crop is enlarged using OpenCV:

```python
cv2.resize(image_plate, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)
```

5. The resized plate is converted to grayscale.
6. Preprocessing is applied to make the characters clearer.
7. PaddleOCR recognizes the text from the processed plate image.

Grayscale conversion is not an enhancement by itself. It removes color information and keeps intensity values, which makes thresholding and contrast-based preprocessing more consistent.

## Preprocessing Methods

The app provides these preprocessing options:

- **CLAHE + Otsu**: Improves local contrast and then applies Otsu thresholding. This worked best in this project for low-contrast plates.
- **Adaptive Gaussian**: Useful when the plate has uneven lighting, shadows, or glare.
- **Sharpen + Otsu**: Useful when the cropped plate is slightly blurry.
- **Otsu**: Works well when the plate already has clean lighting and strong contrast.

## Features

- Streamlit web interface
- YOLO-based number plate detection
- Plate cropping and resizing
- Multiple preprocessing options
- PaddleOCR text recognition
- Output image with detected plate and extracted text

## Tech Stack

- Python
- Streamlit
- Ultralytics YOLO
- PaddleOCR
- OpenCV
- NumPy
- Pillow

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Install dependencies:

```bash
pip install -r requirement.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app_paddle_ocr.py
```

Open the local URL shown in the terminal, upload a vehicle image, choose a preprocessing method, and click **Detect Number Plate**.

## Notes

- YOLO is used only for detecting the number plate location.
- PaddleOCR is used only for recognizing the text.
- OCR accuracy depends heavily on the quality of the cropped and preprocessed plate image.
- `CLAHE + Otsu` is a good first option for most low-contrast number plate images.

## Author

Created by Muhammed Azeem.

