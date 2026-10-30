import numpy as np
import torch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)
yolo = YOLO('yolov8m.pt')

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
dmodel = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()

COCO = {2: 'car', 7: 'truck', 0: 'pedestrian', 5: 'bus', 3: 'motorcycle', 1: 'bicycle'}
REAL_HEIGHT = {'car': 1.6, 'truck': 3.0, 'pedestrian': 1.75,
               'bus': 3.3, 'motorcycle': 1.5, 'bicycle': 1.5}
CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def dav2_map(path):
    img = Image.open(path).convert('RGB')
    with torch.no_grad():
        out = dmodel(**processor(images=img, return_tensors='pt'))
    post = processor.post_process_depth_estimation(
        out, target_sizes=[(img.height, img.width)]
    )
    return post[0]['predicted_depth'].numpy()


def center_patch(box):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return x1 + bw/4, y1 + bh/4, x2 - bw/4, y2 - bh/4


def lidar_ref(u, v, d, box):
    a, b, c, e = center_patch(box)
    m = (u > a) & (u < c) & (v > b) & (v < e)
    return float(np.median(d[m])) if m.sum() >= 5 else None


val = create_splits_scenes()['mini_val']
tokens = []
for s in nusc.scene:
    if s['name'] in val:
        t = s['first_sample_token']
        for _ in range(6):
            tokens.append(t)
            t = nusc.get('sample', t)['next']

rows = []
for tok in tokens:
    sample = nusc.get('sample', tok)
    for cam in CAMS:
        cam_tok = sample['data'][cam]
        cam_data = nusc.get('sample_data', cam_tok)
        path = f"./data/nuscenes/{cam_data['filename']}"
        K = np.array(nusc.get('calibrated_sensor',
                              cam_data['calibrated_sensor_token'])['camera_intrinsic'])

        pts, depths, _ = nusc.explorer.map_pointcloud_to_image(
            sample['data']['LIDAR_TOP'], cam_tok)
        u, v = pts[0], pts[1]

        dmap = dav2_map(path)
        for box in yolo(path, conf=0.25, verbose=False)[0].boxes:
            cls = int(box.cls[0])
            if cls not in COCO:
                continue
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            ref = lidar_ref(u, v, depths, (x1, y1, x2, y2))
            if ref is None:
                continue
            a, b, c, e = [int(x) for x in center_patch((x1, y1, x2, y2))]
            patch = dmap[max(b, 0):e, max(a, 0):c]
            if patch.size == 0:
                continue
            d_dav2 = float(np.median(patch))
            d_heur = K[1, 1] * REAL_HEIGHT[COCO[cls]] / max(y2 - y1, 1.0)
            rows.append((COCO[cls], ref,
                         float(np.clip(d_dav2, 2, 80)),
                         float(np.clip(d_heur, 2, 80))))


ref = np.array([r[1] for r in rows])
dav = np.array([r[2] for r in rows])
heu = np.array([r[3] for r in rows])
print(f"\nPorównanych boxów: {len(rows)}\n")


def report(name, pred, ref):
    ratio = pred / ref
    absrel = np.mean(np.abs(pred - ref) / ref)
    d125 = np.mean(np.maximum(pred / ref, ref / pred) < 1.25) * 100
    print(f"{name:12s} mediana(pred/LiDAR)={np.median(ratio):.2f}  "
          f"AbsRel={absrel:.2f}  w granicach 25%: {d125:.0f}%")


print("=== Wszystkie obiekty ===")
report("Depth Anything", dav, ref)
report("Heurystyka", heu, ref)

for lo, hi in [(0, 15), (15, 30), (30, 80)]:
    m = (ref >= lo) & (ref < hi)
    if m.sum() > 5:
        print(f"\n=== LiDAR {lo}-{hi} m ({m.sum()} boxów) ===")
        report("Depth Anything", dav[m], ref[m])
        report("Heurystyka", heu[m], ref[m])