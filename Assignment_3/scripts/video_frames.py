from pathlib import Path
import cv2
from ultralytics import YOLO

VIDEO_PATH = "chico_fixed.mp4"
DEVICE = "mps"
IMGSZ = 1280

MODELS = {
    "pretrained_yolov8n": "yolov8n.pt",
    "finetuned_6402": "runs/detect/runs/finetune_6402/weights/best.pt",
}

SETTINGS = [
    {"pred_conf": 0.25, "iou": 0.70, "save_conf": 0.50, "tag": "conf025_iou070"},
    {"pred_conf": 0.10, "iou": 0.50, "save_conf": 0.50, "tag": "conf010_iou050"},
]

OUT_BASE = Path("outputs/video_frames")
OUT_BASE.mkdir(parents=True, exist_ok=True)

def run_one(model_name, model_path, pred_conf, iou, save_conf, tag):
    out_dir = OUT_BASE / model_name / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0
    saved = 0

    print(f"\n=== {model_name} | {tag} ===")
    print("Saving to:", out_dir)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1

        r = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=pred_conf,
            iou=iou,
            device=DEVICE,
            verbose=False
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            continue

        confs = r.boxes.conf.detach().cpu().numpy()

        if (confs >= save_conf).any():
            annotated = r.plot()
            t = frame_id / fps
            out_path = out_dir / f"frame_{frame_id:06d}_t{t:.2f}.jpg"
            cv2.imwrite(str(out_path), annotated)
            saved += 1

    cap.release()
    print(f"Saved frames (conf >= {save_conf}): {saved}")

if __name__ == "__main__":
    for model_name, model_path in MODELS.items():
        for s in SETTINGS:
            run_one(
                model_name,
                model_path,
                s["pred_conf"],
                s["iou"],
                s["save_conf"],
                s["tag"],
            )

    print("\nDone.")
    print("Compare results in: outputs/video_frames/")