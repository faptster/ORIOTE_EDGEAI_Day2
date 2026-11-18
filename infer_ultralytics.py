"""CPU-only inference starter. All weights/dependencies must be installed locally."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ['black', 'broken', 'good', 'husk', 'immature', 'parchment', 'shell']
README_NOTES = """Run: python infer.py --input test/ --weights weights/ --out outputs/
Use local best.pt or weights/best_ncnn_model/. All computation uses CPU.
Automatic ROI is a conservative light-paper heuristic. Use --roi full for
other backgrounds, and benchmark accuracy and latency on Raspberry Pi 4.
Masks are returned at original EXIF-corrected resolution, with matching polygons.
Predicted-mask postprocessing explicitly keeps the largest visible component
and fills its internal holes. Changes are reported on stderr. A lower-priority
mask is rejected if hole filling would overwrite a higher-confidence instance.
These are output-format choices, not guarantees of boundary accuracy.
No model or dependency downloads are allowed during this script.
"""

def warn(message):
    print('POSTPROCESS:', message, file=sys.stderr)

def read_rgb(path):
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert('RGB'))

def roi_box(rgb, mode='auto'):
    h, w = rgb.shape[:2]
    full = (0, 0, w, h)
    if mode == 'full':
        return full
    scale = min(1., 1024/max(h, w))
    small = cv2.resize(rgb, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    if np.median(hsv[..., 1]) > 35 or np.median(hsv[..., 2]) < 140:
        return full
    fg = (((hsv[..., 1] > 38) & (hsv[..., 2] < 247)) | (hsv[..., 2] < 80)).astype('uint8')
    n, components, stats, _ = cv2.connectedComponentsWithStats(fg)
    selected = np.zeros_like(fg)
    for k, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        if area < 12:
            continue
        touches = x == 0 or y == 0 or x+bw == fg.shape[1] or y+bh == fg.shape[0]
        if touches and max(bw/bh, bh/bw) > 8:
            continue  # narrow dark page edge; never a central-position crop
        selected[components == k] = 1
    ys, xs = np.where(selected)
    if not len(xs):
        return full
    pad = max(24, .15*max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)/scale)
    box = (max(0, int(xs.min()/scale-pad)), max(0, int(ys.min()/scale-pad)),
           min(w, int((xs.max()+1)/scale+pad)), min(h, int((ys.max()+1)/scale+pad)))
    return full if (box[2]-box[0])*(box[3]-box[1]) > .85*w*h else box

def polygon_mask(raw, occupied, tag='', min_area=16):
    visible = np.asarray(raw, dtype=bool) & ~occupied
    n, labels, stats, _ = cv2.connectedComponentsWithStats(visible.astype('uint8'), connectivity=8)
    if n <= 1:
        warn(f'{tag}: no visible pixels; discarded')
        return None
    k = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    chosen = (labels == k).astype('uint8')
    contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(chosen)
    cv2.fillPoly(filled, [contour], 1)
    if np.any(filled.astype(bool) & occupied):
        warn(f'{tag}: polygon would overwrite another instance; discarded')
        return None
    if filled.sum() < min_area or len(contour) < 3 or cv2.contourArea(contour) < 1:
        warn(f'{tag}: too small for a polygon; discarded')
        return None
    dropped = int(visible.sum()-chosen.sum())
    added = int(filled.sum()-chosen.sum())
    if dropped or added:
        warn(f'{tag}: removed {dropped} disconnected pixels; filled {added} hole pixels')
    return filled.astype(bool), contour.reshape(-1, 2)

def output_instances(masks, scores, classes, shape, offset=(0, 0), names=None):
    names = names or dict(enumerate(CLASSES))
    result = np.zeros(shape, dtype=np.uint16)
    objects = []
    ox, oy = offset
    for index in np.argsort(-np.asarray(scores), kind='stable'):
        m = np.asarray(masks[index], dtype=bool)
        if oy+m.shape[0] > shape[0] or ox+m.shape[1] > shape[1]:
            raise ValueError('Prediction mask extends beyond original image')
        view = result[oy:oy+m.shape[0], ox:ox+m.shape[1]]
        fixed = polygon_mask(m, view != 0, tag=f'prediction {index}')
        if fixed is None:
            continue
        binary, contour = fixed
        instance_id = len(objects)+1
        if instance_id > 65535:
            raise ValueError('Too many instances for uint16')
        view[binary] = instance_id
        objects.append({'id': instance_id, 'class': names[int(classes[index])],
                        'polygon': (contour+np.array([ox, oy])).astype(int).tolist()})
    return result, objects

def local_weights(folder, backend='auto'):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError('--weights must be a local directory')
    pt = folder/'best.pt'
    if backend == 'pt':
        if not pt.is_file(): raise FileNotFoundError('Missing best.pt')
        return pt
    ncnn = sorted(p for p in folder.iterdir() if p.is_dir() and list(p.glob('*.param')) and list(p.glob('*.bin')))
    if len(ncnn) == 1:
        return ncnn[0]
    if len(ncnn) > 1:
        raise ValueError('More than one NCNN export found; keep one model in weights/')
    if backend == 'ncnn':
        raise FileNotFoundError('No single NCNN export found')
    pt = folder/'best.pt'
    if pt.is_file():
        return pt
    raise FileNotFoundError('Expected best.pt or one exported NCNN model subfolder')

def run(args):
    source, destination = Path(args.input).resolve(), Path(args.out).resolve()
    if not source.is_dir():
        raise FileNotFoundError(args.input)
    if source == destination or source in destination.parents:
        raise ValueError('Keep outputs outside the input folder')
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files:
        raise ValueError('No supported input images')
    if len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError('Duplicate input stems would overwrite output PNGs')
    weights = local_weights(args.weights, getattr(args,'backend','auto'))
    config_path = Path(args.weights)/'inference_config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    imgsz = args.imgsz if args.imgsz is not None else int(config.get('imgsz',640))
    conf = args.conf if args.conf is not None else float(config.get('conf',.25))
    if imgsz <= 0 or not 0 <= conf <= 1:
        raise ValueError('Invalid imgsz/conf')
    roi = args.roi if args.roi is not None else config.get('roi','auto')
    max_det = int(config.get('max_det',500))
    if roi not in ('auto','full') or max_det < 1: raise ValueError('Invalid ROI/max_det config')
    started = time.perf_counter()
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    from ultralytics import YOLO
    model = YOLO(str(weights), task='segment')
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    if set(names.values()) != set(CLASSES) or len(names) != 7:
        raise ValueError('Model class names do not match the seven competition classes')
    if args.runtime:
        print(f'Setup (import ultralytics + load {weights.name}): {time.perf_counter()-started:.2f} s')
    for kind in ['seg_mask']+(['vis'] if args.vis else []):
        (destination/kind).mkdir(parents=True, exist_ok=True)
    records = []
    times = []
    for index, path in enumerate(files, 1):
        started = time.perf_counter()
        rgb = read_rgb(path)
        x0, y0, x1, y1 = roi_box(rgb, roi)
        crop = rgb[y0:y1, x0:x1]
        pred = model.predict(Image.fromarray(crop), device='cpu', imgsz=imgsz,
                             conf=conf, max_det=max_det, retina_masks=True, augment=False, verbose=False)[0]
        if pred.masks is None:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint16)
            objects = []
        else:
            masks = pred.masks.data.cpu().numpy() > .5
            if masks.shape[1:] != crop.shape[:2]:
                raise ValueError('Prediction mask is not at native ROI resolution')
            mask, objects = output_instances(masks, pred.boxes.conf.cpu().numpy(),
                pred.boxes.cls.cpu().numpy().astype(int), rgb.shape[:2], (x0, y0), names)
        Image.fromarray(mask).save(destination/'seg_mask'/(path.stem+'.png'))
        records.append({'image_name': path.name, 'objects': objects})
        if args.vis:
            overlay = rgb.copy()
            for obj in objects:
                color = np.random.default_rng(obj['id']).integers(40, 240, 3)
                fg = mask == obj['id']
                overlay[fg] = (.6*rgb[fg]+.4*color).astype('uint8')
            Image.fromarray(overlay).save(destination/'vis'/(path.stem+'.jpg'))
        times.append(time.perf_counter()-started)  # read + ROI + predict + postprocess + save
        if args.runtime:
            print(f'[{index}/{len(files)}] {path.name}: {len(objects)} object(s), {times[-1]:.3f} s')
    (destination/'predictions.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
    if args.runtime:
        rest = times[1:] or times
        print(f'Time per image: mean {sum(times)/len(times):.3f} s, max {max(times):.3f} s, '
              f'total {sum(times):.2f} s for {len(times)} image(s); '
              f'mean without the first (warm-up) image {sum(rest)/len(rest):.3f} s')
    print(f'Saved {len(records)} images to {destination}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['input', 'weights', 'out']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--roi', choices=['auto', 'full'], default=None)
    parser.add_argument('--backend', choices=['auto','pt','ncnn'], default='auto')
    parser.add_argument('--imgsz', type=int, default=None)
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--vis', action='store_true', default=True)
    parser.add_argument('--no-vis', dest='vis', action='store_false')
    parser.add_argument('--runtime', action='store_true', help='print processing time per image')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
