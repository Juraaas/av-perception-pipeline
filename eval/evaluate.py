from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.splits import create_splits_scenes
from ultralytics import YOLO
from pyquaternion import Quaternion
import numpy as np
import json
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import sys
DEPTH_MODE = sys.argv[1] if len(sys.argv) > 1 else 'heuristic'

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)
model = YOLO('yolov8m.pt')

COCO_TO_NUSCENES = {
    2: 'car',
    7: 'truck',
    0: 'pedestrian',
    5: 'bus',
    3: 'motorcycle',
    1: 'bicycle',
}

AV_CLASSES = {2, 7, 0, 5, 3, 1}
CAM_CHANNELS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
depth_model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()


def predict_metric_depth(img_path):
    img = Image.open(img_path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt')
    with torch.no_grad():
        outputs = depth_model(**inputs)
    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(img.height, img.width)])
    return post[0]['predicted_depth'].numpy()


def box_depth(depth_map, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = depth_map.shape
    bw, bh = x2 - x1, y2 - y1
    patch = depth_map[max(y1 + bh//4, 0):min(y2 - bh//4, h),
                      max(x1 + bw//4, 0):min(x2 - bw//4, w)]
    return float(np.median(patch)) if patch.size else None


def get_camera_intrinsics(nusc, sample_token, cam_channel):
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][cam_channel])
    calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    return np.array(calib['camera_intrinsic']), calib, cam_data


def detection_to_nuscenes_box(box_2d, depth, calib, cam_data, category):
    """
    Convert 2D detection + depth to nuScenes 3D box format
    Simplified using 2D box center and constant depth
    """
    x1, y1, x2, y2 = box_2d
    cx_px = (x1 + x2) / 2
    cy_px = (y1 + y2) / 2

    K = np.array(calib['camera_intrinsic'])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Inverse projection: pixel + depth -> camera 3D
    X = (cx_px - cx) * depth / fx
    Y = (cy_px - cy) * depth / fy
    Z = depth
    point_cam = np.array([X, Y, Z])

    # Transfrom: camera -> ego -> world
    R_cam = Quaternion(calib['rotation']).rotation_matrix
    t_cam = np.array(calib['translation'])
    point_ego = R_cam @ point_cam + t_cam

    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    R_ego = Quaternion(ego_pose['rotation']).rotation_matrix
    t_ego = np.array(ego_pose['translation'])
    point_world = R_ego @ point_ego + t_ego

    DEFAULT_SIZE = {
        'car': [1.9, 4.6, 1.7],
        'truck': [2.7, 10.0, 3.5],
        'pedestrian': [0.7, 0.7, 1.8],
        'bus': [2.9, 12.0, 3.8],
        'motorcycle': [0.8, 2.2, 1.5],
        'bicycle': [0.6, 1.7, 1.3],
    }
    size = DEFAULT_SIZE.get(category, [4.0, 2.0, 1.7])

    return {
        'translation': point_world.tolist(),
        'size': size,
        'rotation': ego_pose['rotation'],
    }


def run_detection_on_scene(nusc, scene_idx=0):
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    results = {}

    sample_count = 0
    while sample_token:
        sample = nusc.get('sample', sample_token)
        sample_results = []

        for cam_channel in CAM_CHANNELS:
            K, calib, cam_data = get_camera_intrinsics(nusc, sample_token, cam_channel)
            img_path = f"./data/nuscenes/{cam_data['filename']}"

            depth_map = predict_metric_depth(img_path) if DEPTH_MODE == 'dav2' else None

            detections = model(img_path, conf=0.25, verbose=False)[0]

            for box in detections.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in AV_CLASSES:
                    continue

                score = float(box.conf[0])
                category = COCO_TO_NUSCENES[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].numpy()
                # Depth: estimating from box height
                # Higher box = closer
                REAL_HEIGHT = {'car': 1.6, 'truck': 3.0, 'pedestrian': 1.75, 
                               'bus': 3.3, 'motorcycle': 1.5, 'bicycle': 1.5}

                depth = None
                if DEPTH_MODE == 'dav2':
                    depth = box_depth(depth_map, (x1, y1, x2, y2))
                if depth is None:
                    box_height_px = y2 - y1
                    depth = K[1, 1] * REAL_HEIGHT[category] / max(box_height_px, 1.0)
                depth = float(np.clip(depth, 2.0, 80.0))

                box_3d = detection_to_nuscenes_box((x1, y1, x2, y2), depth, calib, cam_data, category)

                sample_results.append({
                    'sample_token': sample_token,
                    'translation': box_3d['translation'],
                    'size': box_3d['size'],
                    'rotation': box_3d['rotation'],
                    'velocity': [0.0, 0.0],
                    'detection_name': category,
                    'detection_score': score,
                    'attribute_name': '',
                })
        results[sample_token] = sample_results
        sample_count += 1
        print(f"Sample {sample_count}: {len(sample_results)} detections")
        sample_token = sample['next']

    return results


def evaluate(results_dict, output_dir='eval/results'):
    os.makedirs(output_dir, exist_ok=True)

    submission = {
        'meta': {
            'use_camera': True,
            'use_lidar': False,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        },
        'results': results_dict
    }

    results_path = f'{output_dir}/results.json'
    with open(results_path, 'w') as f:
        json.dump(submission, f)
    print(f"Results saved: {results_path}")

    # Official eval
    cfg = config_factory('detection_cvpr_2019')
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=results_path,
        eval_set='mini_val',
        output_dir=output_dir,
        verbose=True
    )

    metrics, metric_data = nusc_eval.evaluate()

    print("\n" + "="*50)
    print("EVAL RESULT")
    print("="*50)
    print(f"mAP: {metrics.mean_ap:.4f}")
    print(f"NDS: {metrics.nd_score:.4f}")
    print("\n Per category (AP):")
    for name, ap in metrics.mean_dist_aps.items():
        print(f" {name:35s}: {ap:.4f}")

    return metrics


val_names = create_splits_scenes()['mini_val']
val_scene_idxs = [i for i, s in enumerate(nusc.scene) if s['name'] in val_names]
print(f"mini_val scenes: {[nusc.scene[i]['name'] for i in val_scene_idxs]}")

results = {}
for idx in val_scene_idxs:
    print(f"Running det on scene {idx} ({nusc.scene[idx]['name']})")
    results.update(run_detection_on_scene(nusc, scene_idx=idx))

print(f"\nNumber of samples: {len(results)}")
print(f"Number of detections: {sum(len(v) for v in results.values())}")

metrics = evaluate(results, output_dir=f'eval/results/{DEPTH_MODE}')
print(metrics.tp_errors)
