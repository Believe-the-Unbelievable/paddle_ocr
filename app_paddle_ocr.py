import cv2
import numpy as np
import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = "ocr.pt"
PREPROCESSING_METHODS = [
    "CLAHE + Otsu",
    "Adaptive Gaussian",
    "Sharpen + Otsu",
    "Otsu",
]


@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_paddle_reader():
    os.environ["PADDLE_PDX_CPU_NUM_THREADS"] = "4"

    from paddleocr import PaddleOCR

    paddle_cpu_config = {
        "paddle_static": {
            "run_mode": "paddle",
            "cpu_threads": 4,
            "enable_new_ir": False,
            "enable_cinn": False,
        }
    }

    try:
        return PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,
            enable_hpi=False,
            cpu_threads=4,
            engine_config=paddle_cpu_config,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        return PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            enable_mkldnn=False,
            cpu_threads=4,
        )


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def trim_plate_border(image_plate, trim_percent=0.03):
    height, width = image_plate.shape[:2]
    trim_x = int(width * trim_percent)
    trim_y = int(height * trim_percent)

    if trim_x == 0 or trim_y == 0:
        return image_plate

    return image_plate[trim_y : height - trim_y, trim_x : width - trim_x]


def clean_binary_image(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def preprocess_plate(image_plate, preprocessing_method, trim_border):
    if trim_border:
        image_plate = trim_plate_border(image_plate)

    resized_new = cv2.resize(
        image_plate,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(resized_new, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if preprocessing_method == "Adaptive Gaussian":
        processed = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
    elif preprocessing_method == "Sharpen + Otsu":
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)
        _, processed = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    elif preprocessing_method == "Otsu":
        _, processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    else:
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, processed = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    processed = clean_binary_image(processed)
    return resized_new, gray, enhanced, processed


def parse_paddle_mapping(data):
    if "res" in data and isinstance(data["res"], dict):
        data = data["res"]

    texts = data.get("rec_texts")
    scores = data.get("rec_scores")

    if texts is None:
        texts = []
    if scores is None:
        scores = []

    text_items = []
    for index, text in enumerate(texts):
        confidence = scores[index] if index < len(scores) else 0.0
        if str(text).strip():
            text_items.append((str(text), float(confidence)))

    return text_items


def parse_paddle_result(result):
    text_items = []

    if not result:
        return text_items

    for page in result:
        if isinstance(page, dict):
            text_items.extend(parse_paddle_mapping(page))
        elif hasattr(page, "json"):
            page_json = page.json() if callable(page.json) else page.json
            if isinstance(page_json, dict):
                text_items.extend(parse_paddle_mapping(page_json))

    if text_items:
        return text_items

    # PaddleOCR 2.x commonly returns [[box, (text, score)], ...] inside a page list.
    page_results = result[0] if len(result) == 1 and isinstance(result[0], list) else result

    for item in page_results:
        if not item or len(item) < 2:
            continue

        text_data = item[1]
        if isinstance(text_data, (list, tuple)) and len(text_data) >= 2:
            text, confidence = text_data[0], text_data[1]
            text_items.append((str(text), float(confidence)))

    return text_items


def read_with_paddle(ocr_image):
    reader = load_paddle_reader()
    paddle_image = cv2.cvtColor(ocr_image, cv2.COLOR_GRAY2BGR)

    if hasattr(reader, "predict"):
        result = reader.predict(paddle_image)
    else:
        result = reader.ocr(paddle_image, cls=True)

    return parse_paddle_result(result)


def detect_number_plates(image_bgr):
    model = load_yolo_model()
    results = model(image_bgr)
    height, width = image_bgr.shape[:2]

    boxes = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = clamp_box(box, width, height)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

    return boxes


def run_pipeline(image_bgr, preprocessing_method, trim_border):
    annotated = image_bgr.copy()
    plate_outputs = []
    all_text = []

    boxes = detect_number_plates(image_bgr)

    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        image_plate = image_bgr[y1:y2, x1:x2]
        resized_new, gray, enhanced, processed = preprocess_plate(
            image_plate,
            preprocessing_method,
            trim_border,
        )

        text_items = read_with_paddle(processed)

        detected_text = " ".join(text for text, _ in text_items).strip()
        confidence_text = ", ".join(
            f"{text} ({confidence:.2f})" for text, confidence in text_items
        )

        if detected_text:
            all_text.append(f"Plate {index}: {detected_text}")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = detected_text if detected_text else "Plate detected"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        plate_outputs.append(
            {
                "number": index,
                "box": (x1, y1, x2, y2),
                "crop": image_plate,
                "resized": resized_new,
                "gray": gray,
                "enhanced": enhanced,
                "processed": processed,
                "text": detected_text,
                "details": confidence_text,
            }
        )

    return annotated, plate_outputs, "\n".join(all_text)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="Number Plate OCR", layout="wide")
st.title("Vehicle Number Plate OCR")

uploaded_file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png"],
)

preprocessing_method = st.selectbox(
    "Choose Preprocessing",
    PREPROCESSING_METHODS,
)

trim_border = st.checkbox(
    "Trim plate border before OCR",
    value=True,
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Number Plate"):
        with st.spinner("Detecting plate and reading text..."):
            try:
                annotated, plate_outputs, extracted_text = run_pipeline(
                    image_bgr,
                    preprocessing_method,
                    trim_border,
                )
            except ModuleNotFoundError as exc:
                st.error(
                    f"{exc.name} is not installed. Install it before using PaddleOCR."
                )
                st.stop()

        if not plate_outputs:
            st.warning("No number plate was detected.")
        else:
            st.image(
                bgr_to_rgb(annotated),
                caption="Detected Number Plate and Recognized Text",
                use_container_width=True,
            )

            st.text_area(
                "Extracted Text",
                extracted_text if extracted_text else "No readable text found.",
                height=140,
            )

            for plate in plate_outputs:
                st.subheader(f"Plate {plate['number']}")
                cols = st.columns(4)
                cols[0].image(
                    bgr_to_rgb(plate["crop"]),
                    caption="Cropped Plate",
                    use_container_width=True,
                )
                cols[1].image(
                    bgr_to_rgb(plate["resized"]),
                    caption="Resized Plate",
                    use_container_width=True,
                )
                cols[2].image(
                    plate["enhanced"],
                    caption="CLAHE Enhanced",
                    use_container_width=True,
                    clamp=True,
                )
                cols[3].image(
                    plate["processed"],
                    caption=f"{preprocessing_method} OCR Input",
                    use_container_width=True,
                    clamp=True,
                )
                st.write(f"Detected text: {plate['text'] or 'No readable text found'}")
                if plate["details"]:
                    st.caption(f"Confidence: {plate['details']}")
