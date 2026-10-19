from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)

AV_CLASSES = {
    2: 'car',
    7: 'truck',
    0: 'person',
    1: 'bicycle',
    3: 'motorcycle',
    5: 'bus',
}

def detect_sample(nusc, sample_token, cam_channel='CAM_FRONT', conf=0.3):
    model = YOLO('yolov8m.pt')
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    cam_data = nusc.get('sample_data', cam_token)
    img_path = f"./data/nuscenes/{cam_data['filename']}"

    results = model(img_path, conf=conf, verbose=False)[0]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.imshow(img)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in AV_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].numpy()
        score = float(box.conf[0])
        label = AV_CLASSES[cls_id]

        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='#e74c3c', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{label} {score:.2f}',
                color='#e74c3c', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
    ax.set_title(f'YOLOv8 detections - {cam_channel}', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'outputs/detections_{cam_channel.lower()}.png', dpi=150)
    plt.show()

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in AV_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].numpy()
        score = float(box.conf[0])
        detections.append([x1, y1, x2, y2, score, cls_id])

    return np.array(detections) if detections else np.zeros((0, 6))


def detect_all_cameras(nusc, sample_token):
    model = YOLO('yolov8m.pt')
    cam_channels = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    sample = nusc.get('sample', sample_token)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    for ax, channel in zip(axes.flatten(), cam_channels):
        cam_token = sample['data'][channel]
        cam_data = nusc.get('sample_data', cam_token)
        img_path = f"./data/nuscenes/{cam_data['filename']}"

        results = model(img_path, conf=0.3, verbose=False)[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax.imshow(img)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in AV_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            score = float(box.conf[0])
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1.5, edgecolor='#e74c3c', facecolor='none'
            )
            ax.add_patch(rect)

        count = sum(1 for b in results.boxes if int(b.cls[0]) in AV_CLASSES)
        ax.set_title(f'{channel} ({count} det.)', fontsize=9)
        ax.axis('off')

    plt.suptitle('YOLOv8 - all cameras', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('outputs/detections_all_cameras.png', dpi=150, bbox_inches='tight')
    plt.show()

sample_token = nusc.sample[0]['token']
dets = detect_sample(nusc, sample_token, 'CAM_FRONT')
print(f"\nAV objects detected: {len(dets)}")
print("Format: [x1, y1, x2, y2, score, class_id]")
print(dets)

detect_all_cameras(nusc, sample_token)