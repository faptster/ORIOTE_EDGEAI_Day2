"""Segment coffee beans with the Day 2 YOLO-seg ONNX model on a Raspberry Pi 4 CPU.

    python infer.py --input test/ --weights weights/ --out outputs/

Writes outputs/seg_mask/<stem>.png (uint16 instance IDs), outputs/predictions.json
(one entry per image with id/class/polygon) and outputs/vis/<stem>.jpg overlays.
"""
import argparse
import ast
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

CLASSES = ("black", "broken", "good", "husk", "immature", "parchment", "shell")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
CONFIG_NAME = "config.json"
DEFAULTS = {
    "model": None,  # ONNX file inside --weights; default best.onnx, else the only *.onnx
    "class_map": {},  # model class name -> one of CLASSES (names already in CLASSES map to themselves)
    "conf": 0.25,  # minimum class score
    "iou": 0.7,  # NMS IoU threshold
    "agnostic_nms": True,  # one bean gives one instance whatever its class
    "max_det": 300,
    "min_area": 20,  # smallest instance kept, in original-image pixels
    "min_keep": 0.5,  # drop an instance when less than this share of its mask is free of earlier ones
}
PAD_VALUE = 114  # Ultralytics letterbox colour
NMS_CLASS_OFFSET = 7680.0  # Ultralytics max_wh for class-aware NMS
CLASS_COLORS = {  # RGB, used for vis/ only
    "black": (35, 35, 35),
    "broken": (230, 60, 60),
    "good": (50, 190, 70),
    "husk": (165, 95, 35),
    "immature": (245, 205, 30),
    "parchment": (60, 135, 245),
    "shell": (200, 70, 220),
}


# ----------------------------------------------------------------------------- model


def load_config(folder):
    config = dict(DEFAULTS)
    path = folder / CONFIG_NAME
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            user = json.load(stream)
        if not isinstance(user, dict):
            raise ValueError(f"{path} must contain a JSON object")
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {', '.join(sorted(unknown))}")
        config.update(user)

    if config["model"] is not None and not isinstance(config["model"], str):
        raise ValueError("config 'model' must be a file name")
    if not isinstance(config["class_map"], dict):
        raise ValueError("config 'class_map' must be an object")
    for key in ("conf", "iou", "min_keep"):
        if not isinstance(config[key], (int, float)) or not 0 <= config[key] <= 1:
            raise ValueError(f"config '{key}' must be a number between 0 and 1")
    for key, low in (("max_det", 1), ("min_area", 3)):
        if not isinstance(config[key], int) or config[key] < low:
            raise ValueError(f"config '{key}' must be an integer >= {low}")
    if not isinstance(config["agnostic_nms"], bool):
        raise ValueError("config 'agnostic_nms' must be true or false")
    return config


def find_model(folder, name):
    if name:
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"Model named in {CONFIG_NAME} not found: {path}")
        return path
    if (folder / "best.onnx").is_file():
        return folder / "best.onnx"
    candidates = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".onnx")
    if not candidates:
        raise FileNotFoundError(f"No .onnx model found in {folder}")
    if len(candidates) > 1:
        raise ValueError(f"Several .onnx models in {folder}. Name one best.onnx or set 'model' in {CONFIG_NAME}.")
    return candidates[0]


class Segmenter:
    def __init__(self, weights):
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        folder = weights if weights.is_dir() else weights.parent
        self.cfg = load_config(folder)
        self.model_path = weights if weights.is_file() else find_model(folder, self.cfg["model"])

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_path), options, providers=["CPUExecutionProvider"]
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        if meta.get("task", "segment") != "segment":
            raise ValueError(f"{self.model_path.name} is a '{meta['task']}' model, not a segmentation model")
        if "names" not in meta:
            raise ValueError(f"{self.model_path.name} has no class names in its metadata")
        names = ast.literal_eval(meta["names"])
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)

        class_map = self.cfg["class_map"]
        self.labels = [class_map.get(name, name) for name in names]
        unmapped = sorted({n for n, label in zip(names, self.labels) if label not in CLASSES})
        if unmapped:
            raise ValueError(
                f"Model classes {unmapped} are not Day 2 classes. Map them in {folder / CONFIG_NAME} "
                f"'class_map' to one of {', '.join(CLASSES)}."
            )

        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(outputs) != 2:
            raise ValueError("Expected a YOLO segmentation model with detection and prototype outputs")
        self.input_name = inputs[0].name
        h, w = inputs[0].shape[2:4]
        if not (isinstance(h, int) and isinstance(w, int)):
            size = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
            h, w = (size, size) if isinstance(size, int) else size
        self.input_hw = (int(h), int(w))

    def detect(self, image):
        """Return [(box_xyxy, class, score, proto_logits)] sorted by score, boxes in image pixels."""
        h0, w0 = image.shape[:2]
        blob, gain, (left, top) = letterbox(image, self.input_hw)
        raw, protos = self.session.run(None, {self.input_name: blob})
        nc, nm = len(self.labels), protos.shape[1]
        if raw.ndim != 3 or raw.shape[1] != 4 + nc + nm:
            raise ValueError(f"Unexpected model output shape {raw.shape}")

        pred = raw[0].T  # (anchors, 4 + nc + nm): cx, cy, w, h, class scores, mask coefficients
        scores = pred[:, 4 : 4 + nc]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(pred)), cls]
        keep = conf > self.cfg["conf"]
        if not keep.any():
            return []
        pred, cls, conf = pred[keep], cls[keep], conf[keep]
        xy, wh = pred[:, :2], pred[:, 2:4]
        boxes = np.concatenate((xy - wh / 2, xy + wh / 2), axis=1)
        order = nms(boxes, conf, self.cfg["iou"], None if self.cfg["agnostic_nms"] else cls)
        order = order[: self.cfg["max_det"]]

        boxes = (boxes[order] - (left, top, left, top)) / gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
        logits = proto_logits(pred[order, 4 + nc :], protos[0], (h0, w0))
        return [
            (boxes[k], self.labels[cls[i]], float(conf[i]), logits[k]) for k, i in enumerate(order)
        ]

    def segment(self, image):
        """Return (uint16 label image, objects, notes). Each object's pixels equal its filled polygon."""
        h0, w0 = image.shape[:2]
        label = np.zeros((h0, w0), np.uint16)
        objects = []
        notes = {"fragment_px": 0, "duplicate": 0, "small": 0, "enclosing": 0}

        for box, name, _, logits in self.detect(image):
            x0, y0, raw = upsample_mask(logits, box, (h0, w0))
            raw_area = int(raw.sum())
            if raw_area == 0:
                notes["small"] += 1
                continue
            region = label[y0 : y0 + raw.shape[0], x0 : x0 + raw.shape[1]]  # view into label
            free = (raw & (region == 0)).astype(np.uint8)  # earlier (higher score) instances keep pixels

            count, components, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
            if count < 2:
                notes["duplicate"] += 1
                continue
            largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
            part = (components == largest).astype(np.uint8)
            contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=len)
            filled = np.zeros_like(part)
            cv2.fillPoly(filled, [contour], 1)  # the component with its holes filled
            filled = filled.astype(bool)

            area = int(filled.sum())
            if area < self.cfg["min_keep"] * raw_area:
                notes["duplicate"] += 1
                continue
            if area < self.cfg["min_area"] or len(contour) < 3:
                notes["small"] += 1
                continue
            if (filled & (region > 0)).any():
                # Another instance sits in a hole of this one; one polygon cannot describe a ring.
                notes["enclosing"] += 1
                continue

            notes["fragment_px"] += int(free.sum()) - int(stats[largest, cv2.CC_STAT_AREA])
            object_id = len(objects) + 1
            if object_id > np.iinfo(np.uint16).max:
                raise ValueError("Too many instances for a uint16 mask")
            region[filled] = object_id
            polygon = (contour[:, 0, :] + (x0, y0)).tolist()
            objects.append({"id": object_id, "class": name, "polygon": polygon})
        return label, objects, notes


# ----------------------------------------------------------------------------- geometry


def letterbox(image, size):
    """Ultralytics-style letterbox: keep aspect ratio, pad with grey to the model input size."""
    h0, w0 = image.shape[:2]
    gain = min(size[0] / h0, size[1] / w0)
    w, h = round(w0 * gain), round(h0 * gain)
    if (w, h) != (w0, h0):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size[1] - w) / 2, (size[0] - h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    blob = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return blob, gain, (left, top)


def nms(boxes, scores, iou_threshold, classes=None):
    if classes is not None:
        boxes = boxes + classes[:, None].astype(boxes.dtype) * NMS_CLASS_OFFSET
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        h = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=int)


def proto_logits(coefs, protos, shape):
    """Mask logits on the prototype grid, cropped to the part that covers the unpadded image."""
    nm, mh, mw = protos.shape
    h0, w0 = shape
    logits = (coefs @ protos.reshape(nm, -1)).reshape(-1, mh, mw)
    gain = min(mh / h0, mw / w0)
    pad_w, pad_h = (mw - w0 * gain) / 2, (mh - h0 * gain) / 2
    top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
    bottom, right = mh - int(round(pad_h + 0.1)), mw - int(round(pad_w + 0.1))
    return logits[:, top:bottom, left:right]


def upsample_mask(logits, box, shape):
    """Bilinear-resize logits to image size (align_corners=False), evaluated inside the box only."""
    h0, w0 = shape
    x0, y0 = int(np.ceil(box[0])), int(np.ceil(box[1]))
    x1, y1 = int(np.ceil(box[2])), int(np.ceil(box[3]))
    if x1 <= x0 or y1 <= y0:
        return x0, y0, np.zeros((0, 0), bool)
    sx, sy = logits.shape[1] / w0, logits.shape[0] / h0
    matrix = np.array([[sx, 0, (x0 + 0.5) * sx - 0.5], [0, sy, (y0 + 0.5) * sy - 0.5]])
    patch = cv2.warpAffine(
        np.ascontiguousarray(logits, dtype=np.float32),
        matrix,
        (x1 - x0, y1 - y0),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return x0, y0, patch > 0


# ----------------------------------------------------------------------------- I/O


def input_images(path):
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = list(path.iterdir())
    else:
        raise FileNotFoundError(f"Input not found: {path}")

    images = sorted(
        (p for p in candidates if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name
    )
    if not images:
        raise ValueError(f"No jpg, jpeg, or png images found: {path}")
    if len({p.name.casefold() for p in images}) != len(images) or len(
        {p.stem.casefold() for p in images}
    ) != len(images):
        raise ValueError("Input file names and stems must be unique (img_01.jpg and img_01.png clash).")
    return images


def read_image(path):
    """RGB uint8 array after applying EXIF orientation."""
    try:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as error:
        raise ValueError(f"Cannot read image: {error}") from error
    return np.array(image)


def check_prediction(label, objects, shape):
    """Fail loudly if the mask and JSON objects do not follow the Day 2 format."""
    h0, w0 = shape
    if label.shape != (h0, w0) or label.dtype != np.uint16:
        raise ValueError("Mask must be single-channel uint16 with the image size")
    ids = [obj["id"] for obj in objects]
    if ids != list(range(1, len(objects) + 1)):
        raise ValueError("Object ids must be 1..N")
    areas = np.bincount(label.ravel(), minlength=len(objects) + 1)
    if len(areas) != len(objects) + 1 or (len(objects) and areas[1:].min() == 0):
        raise ValueError("Mask instance IDs do not match the objects")
    for obj in objects:
        if obj["class"] not in CLASSES:
            raise ValueError(f"Invalid class {obj['class']!r}")
        points = np.array(obj["polygon"], dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"Object {obj['id']} needs a polygon of at least 3 [x, y] points")
        if points.min() < 0 or points[:, 0].max() >= w0 or points[:, 1].max() >= h0:
            raise ValueError(f"Object {obj['id']} polygon leaves the image")
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        canvas = np.zeros((h, w), np.uint8)
        cv2.fillPoly(canvas, [(points - (x, y)).astype(np.int32)], 1)
        drawn = canvas.astype(bool)
        if int(drawn.sum()) != int(areas[obj["id"]]) or not (label[y : y + h, x : x + w][drawn] == obj["id"]).all():
            raise ValueError(f"Object {obj['id']} polygon does not match its mask")


def draw_overlay(image, label, objects):
    vis = image.copy()
    if not objects:
        return vis
    colors = np.zeros((len(objects) + 1, 3), np.float32)
    for obj in objects:
        colors[obj["id"]] = CLASS_COLORS[obj["class"]]
    fg = label > 0
    vis[fg] = (0.55 * vis[fg] + 0.45 * colors[label[fg]]).astype(np.uint8)

    scale = max(0.35, min(image.shape[:2]) / 1500)
    for obj in objects:
        points = np.array(obj["polygon"], np.int32)
        cv2.polylines(vis, [points], True, (255, 255, 255), 1, cv2.LINE_AA)
    for obj in objects:
        x, y = np.array(obj["polygon"]).min(0)
        org = (int(x), max(int(y) - 3, 10))
        text = f"{obj['id']}:{obj['class']}"
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def save_image(path, array, params=()):
    ok, data = cv2.imencode(path.suffix, array, list(params))
    if not ok:
        raise OSError(f"Could not encode {path.name}")
    path.write_bytes(data.tobytes())


def check_replaceable(output):
    """Only replace a directory that holds nothing but results of this script."""
    if not output.is_dir():
        raise ValueError(f"Output path must be a directory: {output}")
    for path in output.rglob("*"):
        rel = path.relative_to(output)
        if rel != Path("predictions.json") and rel.parts[0] not in ("seg_mask", "vis"):
            raise ValueError(f"Output contains other files ({rel}). Use a dedicated directory: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="image file or folder of images")
    parser.add_argument("--weights", type=Path, required=True, help="folder with the ONNX model (+ config.json)")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    args = parser.parse_args()

    source, output, weights = args.input.resolve(), args.out.resolve(), args.weights.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Input and output must be separate.")
    if weights == output or weights in output.parents or output in weights.parents:
        raise ValueError("Weights and output must be separate.")

    images = input_images(source)
    model = Segmenter(weights)
    print(f"Model: {model.model_path.name}, input {model.input_hw[1]}x{model.input_hw[0]}, {len(images)} image(s)")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        check_replaceable(output)

    # Stage a complete run before replacing results from an earlier run.
    with TemporaryDirectory(prefix=".day2-", dir=output.parent) as temp:
        stage = Path(temp) / "new"
        (stage / "seg_mask").mkdir(parents=True)
        (stage / "vis").mkdir()
        predictions = []
        for index, path in enumerate(images, 1):
            try:
                image = read_image(path)
                label, objects, notes = model.segment(image)
                check_prediction(label, objects, image.shape[:2])
                save_image(stage / "seg_mask" / f"{path.stem}.png", label)
                vis = cv2.cvtColor(draw_overlay(image, label, objects), cv2.COLOR_RGB2BGR)
                save_image(stage / "vis" / f"{path.stem}.jpg", vis, (cv2.IMWRITE_JPEG_QUALITY, 90))
            except Exception as error:
                raise RuntimeError(f"{path.name}: {error}") from error
            predictions.append({"image_name": path.name, "objects": objects})

            dropped = [f"{v} {k.replace('_', ' ')}" for k, v in notes.items() if v]
            detail = f" (dropped: {', '.join(dropped)})" if dropped else ""
            print(f"[{index}/{len(images)}] {path.name}: {len(objects)} object(s){detail}")

        with (stage / "predictions.json").open("w", encoding="utf-8") as stream:
            json.dump(predictions, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        if len(list((stage / "seg_mask").glob("*.png"))) != len(images):
            raise RuntimeError("Mask count does not match the number of input images")

        previous = Path(temp) / "previous"
        if output.exists():
            output.rename(previous)
        try:
            stage.rename(output)
        except OSError:
            if previous.exists():
                previous.rename(output)
            raise

    print(f"Done. Results: {output / 'predictions.json'}, {output / 'seg_mask'}, {output / 'vis'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
