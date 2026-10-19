from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
import cv2

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)

def project_lidar_to_camera(nusc, sample_token, cam_channel='CAM_FRONT'):
    sample = nusc.get('sample', sample_token)

    cam_token = sample['data'][cam_channel]
    lidar_token = sample['data']['LIDAR_TOP']

    cam_data = nusc.get('sample_data', cam_token)
    lidar_data = nusc.get('sample_data', lidar_token)

    lidar_path = f"./data/nuscenes/{lidar_data['filename']}"
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points = points[:, :3]

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

    img = cv2.imread(f"./data/nuscenes/{cam_data['filename']}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    u = points_2d[:, 0]
    v = points_2d[:, 1]
    depth = points_cam[:, 2]
    in_frame = (u > 0) & (u < w) & (v > 0) & (v < h)

    u, v, depth = u[in_frame], v[in_frame], depth[in_frame]

    plt.figure(figsize=(14, 6))
    plt.imshow(img)
    plt.scatter(u, v, c=depth, cmap='plasma', s=1.5, alpha=0.7)
    plt.colorbar(label='Depth (m)')
    plt.title(f'LiDAR projected onto {cam_channel}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'outputs/lidar_{cam_channel.lower()}.png', dpi=150)
    plt.show()
    print(f"Punktów w kadrze {len(u)} / {mask.sum()} total")

sample_token = nusc.sample[0]['token']
project_lidar_to_camera(nusc, sample_token, 'CAM_FRONT')
project_lidar_to_camera(nusc, sample_token, 'CAM_FRONT_LEFT')
project_lidar_to_camera(nusc, sample_token, 'CAM_BACK')