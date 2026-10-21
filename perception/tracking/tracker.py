from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)
AV_CLASSES = {2: 'car', 7: 'truck', 0: 'person', 1: 'bicycle', 3: 'motorcycle', 5: 'bus'}

TRACK_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#e91e63'
]

def get_camera_frames(nusc, scene_idx=0, cam_channel='CAM_FRONT', max_frames=10):
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    
    frames = []
    while sample_token and len(frames) < max_frames:
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data'][cam_channel]
        cam_data = nusc.get('sample_data', cam_token)
        frames.append({
            'path': f"./data/nuscenes/{cam_data['filename']}",
            'token': sample_token,
            'timestamp': cam_data['timestamp']
        })
        sample_token = sample['next']

    return frames


def run_tracking(scene_idx=0, cam_channel='CAM_FRONT', max_frames=10):
    model = YOLO('yolov8m.pt')
    frames = get_camera_frames(nusc, scene_idx, cam_channel, max_frames)
    print(f"Tracking {len(frames)} frames from {cam_channel}")

    trajectories = {}
    output_frames = []

    for frame_idx, frame in enumerate(frames):
        img = cv2.imread(frame['path'])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ByteTrack przez persist=True — ultralytics zapamiętuje stan między klatkami
        results = model.track(frame['path'], persist=True,conf=0.3, 
                              iou=0.5, verbose=False, classes=list(AV_CLASSES.keys()))[0]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(img_rgb)

        if results.boxes.id is not None:
            track_ids = results.boxes.id.numpy().astype(int)
            boxes_xyxy = results.boxes.xyxy.numpy()
            classes = results.boxes.cls.numpy().astype(int)
            scores = results.boxes.conf.numpy()

            for tid, box, cls_id, score in zip(track_ids, boxes_xyxy, classes, scores):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                color = TRACK_COLORS[tid % len(TRACK_COLORS)]

                if tid not in trajectories:
                    trajectories[tid] = []
                trajectories[tid].append((cx, cy))

                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

                label = f"ID:{tid} {AV_CLASSES.get(cls_id, '?')} {score:.2f}"
                ax.text(x1, y1-6, label, color='white', fontsize=8,fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor=color, alpha=0.8))

                if len(trajectories[tid]) > 1:
                    traj = np.array(trajectories[tid])
                    ax.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5, alpha=0.4)
                    ax.scatter(traj[:-1, 0], traj[:-1, 1], s=8, color=color, alpha=0.4)
        ax.set_title(f'{cam_channel} - frame {frame_idx+1}/{len(frames)}', fontsize=11)
        ax.axis('off')
        plt.tight_layout()

        out_path = f'outputs/tracking_frame{frame_idx:03d}.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        output_frames.append(out_path)
        plt.close()
        print(f"Frame {frame_idx+1}: {len(trajectories)} unique tracks")

    make_gif(output_frames)
    return trajectories


def make_gif(frame_paths, output='outputs/tracking_demo.gif'):
    from PIL import Image
    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=500, loop=0)
    print(f"\nGIF saved: {output}")


trajectories = run_tracking(scene_idx=0, cam_channel='CAM_FRONT', max_frames=8)
print(f"\nSummary")
print(f"Unique tracking objects: {len(trajectories)}")
for tid, traj in sorted(trajectories.items()):
    print(f"Track ID {tid}: {len(traj)} frames")