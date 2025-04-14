import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

model = YOLO("DetectTumor/yolo_train5/weights/best.pt")
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def predict_and_draw(img: Image.Image):
    results = model.predict(img, imgsz=299, conf=0.25)
    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return img, "No tumor detected."

    img_np = np.array(img).copy()
    report_lines = []

    all_no_tumor = all(int(box.cls[0]) == 2 for box in boxes)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[class_id]} ({conf*100:.2f}%)"
        color = (0, 255, 0) if class_id != 2 else (0, 0, 255)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_np, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)

        if class_id == 2:
            report_lines.append(label)
        else:
            report_lines.append(f"{label} - Box: ({x1}, {y1}), ({x2}, {y2})")

    if all_no_tumor:
        report_text = f"No Tumor detected ({report_lines[0].split('(')[-1]}"
    else:
        report_text = f"{len(boxes)} tumor(s) detected:\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(report_lines))

    return Image.fromarray(img_np), report_text

gr.Interface(
    fn=predict_and_draw,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[
        gr.Image(type="pil", label="Detection Result"),
        gr.Text(label="Detection Report")
    ],
    title="🧠 Brain Tumor Detection",
    description="Detect and localize brain tumors from MRI scans using a custom-trained YOLOv8 model.",
    allow_flagging="never",
    theme="soft"
).launch()
