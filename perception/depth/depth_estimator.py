from transformers import pipeline as hf_pipeline
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes/', verbose=False)
os.makedirs('outputs/depth', exist_ok=True)

print("Load Depth Anything v2")
depth_pipe = hf_pipeline(
    task="depth-estimation", 
    model="depth-anything/Depth-Anything-V2-Small-hf"
)
print("Model loaded")


def get_lidar_depths_for_camera(nusc, sample_token, cam_channel='CAM_FRONT'):
    """
    Returns LiDAR points projected onto the camera along with their true depth.
    """
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][cam_channel]
    lidar_token = sample['data']['LIDAR_TOP']
    cam_data = nusc.get('sample_data', cam_token)
    lidar_data = nusc.get('sample_data', lidar_token)

    lidar_path = f"./data/nuscenes/{lidar_data['filename']}"
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]

    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_ego = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    cam_ego = nusc.get('ego_pose', cam_data['ego_pose_token'])

    R_lidar = Quaternion(lidar_calib['rotation']).rotation_matrix
    t_lidar = np.array(lidar_calib['translation'])
    points_ego = (R_lidar @ points.T).T + t_lidar

    R_ego_lidar = Quaternion(lidar_ego['rotation']).rotation_matrix
    t_ego_lidar = np.array(lidar_ego['translation'])
    points_world = (R_ego_lidar @ points_ego.T).T + t_ego_lidar

    R_ego_cam = Quaternion(cam_ego['rotation']).rotation_matrix
    t_ego_cam = np.array(cam_ego['translation'])
    points_ego_cam = (R_ego_cam.T @ (points_world - t_ego_cam).T).T

    R_cam = Quaternion(cam_calib['rotation']).rotation_matrix
    t_cam = np.array(cam_calib['translation'])
    points_cam = (R_cam.T @ (points_ego_cam - t_cam).T).T

    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]

    K = np.array(cam_calib['camera_intrinsic'])
    points_2d = (K @ points_cam.T).T
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]

    img_path = f"./data/nuscenes/{cam_data['filename']}"
    img = Image.open(img_path)
    w, h = img.size

    u = points_2d[:, 0]
    v = points_2d[:, 1]
    depth_gt = points_cam[:, 2]

    in_frame = (u > 0) & (u < w) & (v > 0) & (v < h)

    return u[in_frame], v[in_frame], depth_gt[in_frame], img_path, img


def estimate_and_compare(nusc, sample_token, cam_channel='CAM_FRONT'):
    """
    Main component: depth estimation + comparison with LiDAR GT
    """
    u, v, depth_gt, img_path, img = get_lidar_depths_for_camera(nusc, sample_token, cam_channel)

    result = depth_pipe(img)
    depth_pred = depth_pred.max() - depth_pred

    # Depth Anything returns RELATIVE depth (not in meters)
    # Normalizing to [0, 1] for vis
    depth_pred_norm = (depth_pred - depth_pred.min() / \
                       (depth_pred.max() - depth_pred.min()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.imshow(img)
    sc = ax.scatter(u, v, c=depth_gt, cmap='plasma', s=2, alpha=0.8, vmin=0, vmax=50)
    plt.colorbar(sc, ax=ax, label='LiDAR depth (m)')
    ax.set_title('LiDAR Ground Truth', fontsize=11)
    ax.axis('off')

    ax = axes[1]
    im = ax.imshow(depth_pred_norm, cmap='plasma')
    plt.colorbar(im, ax=ax, label='Relative depth')
    ax.set_title('Depth Anything v2 (relative)', fontsize=11)
    ax.axis('off')

    # comparison - scatter plot GT vs predicted
    ax = axes[2]
    h_img, w_img = depth_pred.shape

    # Retrieve predictions for pixels where LiDAR GT available
    u_int = np.clip(u.astype(int), 0, w_img - 1)
    v_int = np.clip(v.astype(int), 0, h_img - 1)
    depth_pred_at_lidar = depth_pred_norm[v_int, u_int]

    # Scatter: GT (X axis) vs pred (Y axis)
    ax.scatter(depth_gt, depth_pred_at_lidar, s=3, alpha=0.3, color='steelblue')
    ax.set_xlabel('Lidar depth GT (m)')
    ax.set_ylabel('Model prediction (relative, normalized)')
    ax.set_title('GT vs Predicted depth', fontsize=11)
    ax.grid(alpha=0.3)

    # Trendline, is the prediction monotonic with GT?
    from numpy.polynomial import polynomial as P
    mask_valid = depth_gt < 50
    if mask_valid.sum() > 10:
        coeffs = np.polyfit(depth_gt[mask_valid], depth_pred_at_lidar[mask_valid], 1)
        x_line = np.linspace(0, 50, 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), 'r-', linewidth=2, label=f'trend')
        ax.legend()

    plt.suptitle(f'Depth Estimation - {cam_channel}', fontsize=13, y=1.01)
    plt.subplots_adjust(top=0.88)
    plt.tight_layout()
    out_path = f'outputs/depth/depth_{cam_channel.lower()}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}")

    # --- Metrics ---
    # Depth Anything returns a relative depth, so scale it before calculating the metrics
    # Find the scale using the median ratio
    pred_at_lidar = depth_pred[v_int, u_int]
    scale = np.median(depth_gt[mask_valid] / (pred_at_lidar[mask_valid] + 1e-8))
    depth_pred_scaled = pred_at_lidar * scale

    abs_rel = np.mean(np.abs(depth_gt - depth_pred_scaled) / (depth_gt + 1e-8))
    rmse = np.sqrt(np.mean((depth_gt - depth_pred_scaled) ** 2))

    print(f"\n--- Metrics for {cam_channel} ---")
    print(f"LiDAR points in the frame: {len(depth_gt)}")
    print(f"AbsRel (lower = better): {abs_rel:.3f}")
    print(f"RMSE   (lower = better): {rmse:.3f} m")
    print(f"Calibration scale: {scale:.3f}")

    return abs_rel, rmse


sample_token = nusc.sample[0]['token']
abs_rel, rmse = estimate_and_compare(nusc, sample_token, 'CAM_FRONT')