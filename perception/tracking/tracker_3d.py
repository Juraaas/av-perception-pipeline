from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import sys
sys.path.append('.')
from perception.fusion.fusion import (
    get_lidar_points_in_camera, fit_3d_box, AV_CLASSES, CLASS_COLORS
)

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)
model = YOLO('yolov8m.pt')
os.makedirs('outputs/tracking_3d', exist_ok=True)


def make_kalman_3d():
    """
    State: [x, z, vx, vz]  ← position and velocity in the BEV plane
    Measurement: [x, z]    ← observe position (from fusion)
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 0.5 # nuScenes: ~2Hz between keyframes

    # Model ruchu: pozycja += prędkość * dt
    kf.F = np.array([
        [1, 0, dt, 0],  # x_new = x + vx*dt
        [0, 1, 0, dt],  # z_new = z + vz*dt
        [0, 0, 1, 0],   # vx constant
        [0, 0, 0, 1],   # vz constant
    ])

    # observing x and z (not velocity)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    # Measurement noise, how inaccurate is Fusion (m²)
    kf.R = np.eye(2) * 2.0

    # Process noise - the car may change speed
    kf.Q = np.eye(4) * 0.5
    kf.Q[2, 2] = 1.0 # vx may change more rapidly
    kf.Q[3, 3] = 1.0 # vz may change more rapidly

    # Initial covariance - high uncertainty at the outset
    kf.P = np.eye(4) * 100

    return kf


class Track3D:
    _next_id = 1

    def __init__(self, x, z, label):
        self.id = Track3D._next_id
        Track3D._next_id += 1
        self.label = label
        self.kf = make_kalman_3d()
        self.kf.x = np.array([x, z, 0, 0]) # initialise position, velocity=0
        self.history = [(x, z)] # trajectory
        self.hits = 1 # how many times detected
        self.misses = 0 # how many frames without det
        self.last_box = None  # last 3D box

    def predict(self):
        """The Kalman prediction step — where the object will be."""
        self.kf.predict()
        return self.kf.x[0], self.kf.x[1] # predicted x, z

    def update(self, x, z, box_3d):
        """Update step - we can see the object here."""
        self.kf.update(np.array([x, z]))
        self.history.append((self.kf.x[0], self.kf.x[1]))
        self.hits += 1
        self.misses = 0
        self.last_box = box_3d

    @property
    def position(self):
        return self.kf.x[0], self.kf.x[1] # current position

    @property
    def velocity(self):
        return self.kf.x[2], self.kf.x[3] # current velocity


def iou_bev(box_a, box_b):
    """Simplified IoU in BEV, uses the distance between centres. """
    dist = np.sqrt((box_a[0] - box_b[0])**2 + (box_a[1] - box_b[1])**2)
    return max(0, 1 - dist / 15.0) # 15m = maximum distance threshold


def associate_detections_to_tracks(tracks, detections, threshold=0.2):
    """
    Assign detections to existing tracks using greedy matching.
    For each detection, find the nearest track (if it is close enough).
    """
    if not tracks or not detections:
        return {}, set(range(len(detections)))

    matched = {} # {det_idx: track_idx}
    unmatched_dets = set(range(len(detections)))

    # For each track predict its position and find the nearest detection
    for t_idx, track in enumerate(tracks):
        pred_x, pred_z = track.predict()
        best_score = threshold
        best_det = None

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            score = iou_bev((pred_x, pred_z), (det['center'][0], det['center'][2]))
            if score > best_score and det['label'] == track.label:
                best_score = score
                best_det = d_idx

        if best_det is not None:
            matched[best_det] = t_idx
            unmatched_dets.discard(best_det)

    return matched, unmatched_dets


class Tracker3D:
    def __init__(self, max_misses=3):
        self.tracks = []
        self.max_misses = max_misses

    def update(self, detections):
        """
        Input: a list of 3D boxes from Fusion
        Output: a list of active tracks with their IDs and positions
        """
        # 1. Assign detection to track
        matched, unmatched = associate_detections_to_tracks(self.tracks, detections)

        # 2. Analyse matches
        for det_idx, track_idx in matched.items():
            det = detections[det_idx]
            cx, cz = det['center'][0], det['center'][2]
            self.tracks[track_idx].update(cx, cz, det)

        # 3. Increase misses for unmatched
        matched_tracks = set(matched.values())
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.misses += 1

        # 4. Create new tracks for unmatched dets
        for det_idx in unmatched:
            det = detections[det_idx]
            cx, cz = det['center'][0], det['center'][2]
            self.tracks.append(Track3D(cx, cz, det['label']))
            self.tracks[-1].last_box = det

        # 5. Delete lost tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return self.tracks


def run_3d_tracking(scene_idx=0, cam_channel='CAM_FRONT', max_frames=8):
    tracker = Tracker3D(max_misses=2)
    scene = nusc.scene[scene_idx]
    sample_token = scene['first_sample_token']
    Track3D._next_id = 1

    frame_idx = 0
    output_frames = []

    while sample_token and frame_idx < max_frames:
        sample = nusc.get('sample', sample_token)
        print(f"\nFrame {frame_idx + 1}/{max_frames}")

        pts_cam, u_lidar, v_lidar, img = get_lidar_points_in_camera(nusc, sample_token, cam_channel)
        img_path = f"./data/nuscenes/{nusc.get('sample_data', sample['data'][cam_channel])['filename']}"
        results = model(img_path, conf=0.3, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in AV_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            inside = ((u_lidar >= x1) & (u_lidar <= x2) &
                      (v_lidar >= y1) & (v_lidar <= y2))
            pts_inside = pts_cam[inside]
            if len(pts_inside) < 3:
                continue
            box_3d = fit_3d_box(pts_inside)
            if box_3d:
                box_3d['label'] = AV_CLASSES[cls_id]
                detections.append(box_3d)

        active_tracks = tracker.update(detections)
        print(f"Detections: {len(detections)}, Active tracks: {len(active_tracks)}")

        out_path = visualize_bev_tracking(pts_cam, active_tracks, frame_idx, cam_channel)
        output_frames.append(out_path)
        sample_token = sample['next']
        frame_idx += 1

    make_gif(output_frames, 'outputs/tracking_3d/bev_tracking.gif')
    print(f"\nGIF saved: outputs/tracking_3d/bev_tracking.gif")

def visualize_bev_tracking(pts_cam, tracks, frame_idx, cam_channel):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#0d0d1a')
    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 50)
    ax.set_xlabel('Lateral (m)', color='white')
    ax.set_ylabel('Forward (m)', color='white')
    ax.set_title(f'3D BEV Tracking — {cam_channel} | '
                 f'Frame {frame_idx + 1}', color='white', fontsize=12)
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334466')

    for r in [10, 20, 30, 40]:
        circle = plt.Circle((0, 0), r, fill=False, color='#334466', linewidth=0.8, linestyle='--')
        ax.add_patch(circle)
        ax.text(0.3, r+0.3, f'{r}m', color='#556688', fontsize=8)

    ax.scatter(pts_cam[:, 0], pts_cam[:, 2], s=0.8, alpha=0.2, color='#446688')

    ego = patches.Rectangle((-1, 0), 2, 4, color='white', alpha=0.9, zorder=5)
    ax.add_patch(ego)
    ax.text(0, 2, 'EGO', ha='center', va='center', color='black', fontsize=8, fontweight='bold', zorder=6)

    for track in tracks:
        if track.last_box is None:
            continue

        color = CLASS_COLORS.get(track.label, '#ffffff')
        px, pz = track.position
        vx, vz = track.velocity

        # Trajcectory history
        if len(track.history) > 1:
            hist = np.array(track.history)
            ax.plot(hist[:, 0], hist[:, 1], '-', color=color, linewidth=1.5, alpha=0.6)
            ax.scatter(hist[:-1, 0], hist[:-1, 1], s=10, color=color, alpha=0.4)

        b = track.last_box
        l, w = b.get('length', 4), b.get('width', 2)
        yaw = b.get('yaw', 0)

        corners = np.array([[-l/2,-w/2],[l/2,-w/2],[l/2,w/2],[-l/2,w/2]])
        R_yaw = np.array([[np.cos(yaw),-np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
        corners_rot = (R_yaw @ corners.T).T + np.array([px, pz])

        poly = patches.Polygon(corners_rot, closed=True, linewidth=2, edgecolor=color,
                               facecolor=color, alpha=0.25)
        ax.add_patch(poly)

        # speed vector
        speed = np.sqrt(vx**2 + vz**2)
        if speed > 0.1:
            ax.annotate('', xy=(px + vx*2, pz + vz*2), xytext=(px, pz),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))

        ax.text(px, pz + w/2 + 0.8,
                f"ID:{track.id} {track.label}\n"
                f"{pz:.1f}m | {speed:.1f}m/s",
                color=color, fontsize=7, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='#0d0d1a', alpha=0.7))
    ax.grid(color='#1a1a3e', alpha=0.4)
    plt.tight_layout()
    out_path = f'outputs/tracking_3d/frame_{frame_idx:03d}.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0d0d1a')
    plt.close()
    return out_path

def make_gif(frame_paths, output):
    from PIL import Image as PILImage
    frames = [PILImage.open(p) for p in frame_paths]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=600, loop=0)


run_3d_tracking(scene_idx=0, cam_channel='CAM_FRONT', max_frames=8)