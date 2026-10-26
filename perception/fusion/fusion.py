from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)
model = YOLO('yolov8m.pt')
AV_CLASSES = {2: 'car', 7: 'truck', 0: 'person', 1: 'bicycle', 3: 'motorcycle', 5: 'bus'}
CLASS_COLORS = {'car': '#3498db', 'truck': '#e74c3c', 'person': '#2ecc71',
    'bicycle': '#f39c12', 'motorcycle': '#9b59b6', 'bus': '#1abc9c'}


def get_lidar_points_in_camera(nusc, sample_token, cam_channel='CAM_FRONT'):
    sample = nusc.get('sample', sample_token)
    cam_data = nusc.get('sample_data', sample['data'][cam_channel])
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

    points = np.fromfile(f"./data/nuscenes/{lidar_data['filename']}", 
                         dtype=np.float32).reshape(-1, 5)[:, :3]

    lc = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    cc = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    le = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ce = nusc.get('ego_pose', cam_data['ego_pose_token'])

    pts = (Quaternion(lc['rotation']).rotation_matrix @ points.T).T + lc['translation']
    pts = (Quaternion(le['rotation']).rotation_matrix @ pts.T).T + le['translation']
    pts = (Quaternion(ce['rotation']).rotation_matrix.T @ (pts - ce['translation']).T).T
    pts_cam = (Quaternion(cc['rotation']).rotation_matrix.T @ (pts - cc['translation']).T).T

    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]

    K = np.array(cc['camera_intrinsic'])
    p2d = (K @ pts_cam.T).T
    u = p2d[:, 0] / p2d[:, 2]
    v = p2d[:, 1] / p2d[:, 2]

    img = Image.open(f"./data/nuscenes/{cam_data['filename']}")
    w, h = img.size
    in_frame = (u > 0) & (u < w) & (v > 0) & (v < h)

    return pts_cam[in_frame], u[in_frame], v[in_frame], img


def fit_3d_box(points_3d):
    """
    Fit 3D bounding box to point cloud through PCA.
    Returns: center (x,y,z), dimensions (l,w,h), yaw angle
    """
    if len(points_3d) < 3:
        return None

    center = points_3d.mean(axis=0)
    centered = points_3d - center

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt[0] = first component = the object’s longitudinal axis (e.g. along the length of a car)
    # Vt[1] = second component = the transverse axis
    main_axis = Vt[0]  # direction of the object’s orientation

    # Yaw angle — rotation in the XY plane (top view)
    yaw = np.arctan2(main_axis[1], main_axis[0])

    # Project the points onto the PCA axes to obtain the dimensions
    proj_main = centered @ Vt[0]
    proj_side = centered @ Vt[1]
    proj_up = centered @ Vt[2]

    length = proj_main.max() - proj_main.min()
    width = proj_side.max() - proj_side.min()
    height = proj_up.max() - proj_up.min()

    return {
        'center': center,
        'length': length,
        'width': width,
        'height': height,
        'yaw': yaw,
        'n_points': len(points_3d)
    }


def run_fusion(nusc, sample_token, cam_channel='CAM_FRONT'):
    """
    2D det + LiDAR → 3D boxes + BEV.
    """
    pts_cam, u_lidar, v_lidar, img = get_lidar_points_in_camera(nusc, sample_token, cam_channel)

    img_path = f"./data/nuscenes/" \
    f"{nusc.get('sample_data', nusc.get('sample', sample_token)['data'][cam_channel])['filename']}"
    results = model(img_path, conf=0.3, verbose=False)[0]

    # For each det - take out LiDAR points in box center
    boxes_3d = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in AV_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].numpy()
        score = float(box.conf[0])
        label = AV_CLASSES[cls_id]

        inside = (
            (u_lidar >= x1) & (u_lidar <= x2) &
            (v_lidar >= y1) & (v_lidar <= y2)
        )
        pts_inside = pts_cam[inside]

        if len(pts_inside) < 3:
            continue

        # Fit 3D box through PCA
        box_3d = fit_3d_box(pts_inside)
        if box_3d:
            box_3d['label'] = label
            box_3d['score'] = score
            box_3d['box_2d'] = (x1, y1, x2, y2)
            boxes_3d.append(box_3d)

    visualize_fusion(img, pts_cam, u_lidar, v_lidar, boxes_3d, cam_channel, sample_token)
    return boxes_3d


def visualize_fusion(img, pts_cam, u_lidar, v_lidar, boxes_3d, cam_channel, sample_token):
    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)

    depth = pts_cam[:, 2]
    ax1.scatter(u_lidar, v_lidar, c=depth, cmap='plasma', s=1.5, alpha=0.5, vmin=0, vmax=50)

    for b in boxes_3d:
        x1, y1, x2, y2 = b['box_2d']
        color = CLASS_COLORS.get(b['label'], 'white')
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1 - 5,
                 f"{b['label']} {b['score']:.2f} | "
                 f"d={b['center'][2]:.1f}m | pts={b['n_points']}",
                 color='white', fontsize=8, fontweight='bold', 
                 bbox=dict(facecolor=color, alpha=0.7, pad=2))
        ax1.set_title(f'Camera + LiDAR fusion - {cam_channel}', fontsize=11)
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_facecolor('#1a1a2e')
        ax2.set_xlim(-20, 20)
        ax2.set_ylim(0, 50)
        ax2.set_xlabel('Lateral (m)', color='white')
        ax2.set_ylabel('Forward (m)', color='white')
        ax2.set_title('Bird\'s Eye View (BEV)', fontsize=11, color='white')
        ax2.tick_params(colors='white')
        ax2.grid(color='#333355', alpha=0.5)

        ego = patches.Rectangle((-1, -2), 2, 4, color='white', alpha=0.9, zorder=5)
        ax2.add_patch(ego)
        ax2.text(0, -1, 'EGO', ha='center', va='center', color='black', fontsize=8, fontweight='bold', zorder=6)

        for r in [10, 20, 30, 40]:
            circle = plt.Circle((0, 0), r, fill=False, color='#334466', linewidth=0.8)
            ax2.add_patch(circle)
            ax2.text(r + 0.3, 0.5, f'{r}m', color='#556688', fontsize=7)

        ax2.scatter(pts_cam[:, 0], pts_cam[:, 2], s=0.5, alpha=0.3, color='#446688')

        for b in boxes_3d:
            cx, cy, cz = b['center']
            color = CLASS_COLORS.get(b['label'], 'white')

            l, w = b['length'], b['width']
            yaw = b['yaw']

            corners = np.array([
                [-l/2, -w/2],
                [ l/2, -w/2],
                [ l/2,  w/2],
                [-l/2,  w/2],
            ])

            R_yaw = np.array([[np.cos(yaw), -np.sin(yaw)],
                              [np.sin(yaw), np.cos(yaw)]])
            corners_rot = (R_yaw @ corners.T).T
            corners_bev = corners_rot + np.array([cx, cz])

            poly = patches.Polygon(corners_bev, closed=True, linewidth=2,
                                   edgecolor=color, facecolor=color, alpha=0.3)
            ax2.add_patch(poly)

            arrow_len = max(l, 1.5)
            dx = np.cos(yaw) * arrow_len
            dz = np.sin(yaw) * arrow_len
            ax2.annotate('', xy=(cx + dx, cz + dz), xytext=(cx, cz), 
                         arrowprops=dict(arrowstyle='->', color=color, lw=2))
            ax2.text(cx, cz + w/2 + 0.5,
                     f"{b['label']}\n{cz:.1f}m",
                     color=color, fontsize=7, ha='center', fontweight='bold')

        for spine in ax2.spines.values():
            spine.set_edgecolor('#334466')

        plt.tight_layout()
        plt.savefig('outputs/fusion_bev.png', dpi=150,
                    bbox_inches='tight', facecolor='#0d0d1a')
        plt.show()
        print(f"\nDetected 3D objects: {len(boxes_3d)}")
        for b in boxes_3d:
            print(f"  {b['label']:10s} | "
                f"dist={b['center'][2]:.1f}m | "
                f"L={b['length']:.1f}m W={b['width']:.1f}m | "
                f"pts={b['n_points']}")


sample_token = nusc.sample[0]['token']
boxes_3d = run_fusion(nusc, sample_token, 'CAM_FRONT')