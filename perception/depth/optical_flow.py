from nuscenes.nuscenes import NuScenes
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=False)

def get_consecutive_frames(nusc, scene_idx=0, cam_channel='CAM_FRONT'):
    scene = nusc.scene[scene_idx]
    sample = nusc.get('sample', scene['first_sample_token'])

    frames = []
    for _ in range(2):
        cam_token = sample['data'][cam_channel]
        cam_data = nusc.get('sample_data', cam_token)
        frames.append(f"./data/nuscenes/{cam_data['filename']}")
        if sample['next']:
            sample = nusc.get('sample', sample['next'])

    return frames[0], frames[1]


def run_optical_flow(scene_idx=0, cam_channel='CAM_FRONT'):
    path1, path2 = get_consecutive_frames(nusc, scene_idx, cam_channel)

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print(f"Frame 1: {path1.split('/')[-1]}")
    print(f"Frame 2: {path2.split('/')[-1]}")
    print(f"Size: {img1.shape}")

    print("\nLoading RAFT")
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights)
    model.eval()

    # Preprocessing - RAFT requires float32 tensors in the range [0,1]
    def preprocess(img_rgb):
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        # RAFT requires dimensions that are divisible by 8
        h, w = t.shape[1], t.shape[2]
        h_new = (h // 8) * 8
        w_new = (w // 8) * 8
        t = t[:, :h_new, :w_new]
        return t.unsqueeze(0) # add batch dimension

    t1 = preprocess(img1_rgb)
    t2 = preprocess(img2_rgb)

    print("Computing Optical flow")
    with torch.no_grad():
        flow_list = model(t1, t2)
    flow = flow_list[-1].squeeze(0).numpy() # (2, H, W) - flow_x i flow_y

    print(f"Flow shape: {flow.shape}")
    print(f"Flow range X: [{flow[0].min():.1f}, {flow[0].max():.1f}] px")
    print(f"Flow range Y: [{flow[1].min():.1f}, {flow[1].max():.1f}] px")

    visualize_flow(img1_rgb, img2_rgb, flow, cam_channel)

    return flow


def visualize_flow(img1, img2, flow, cam_channel):
    h, w = flow.shape[1], flow.shape[2]

    # Flow magnitude - how fast each pixel moves
    flow_magnitude = np.sqrt(flow[0]**2 + flow[1]**2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].imshow(img1[:h, :w])
    axes[0, 0].set_title('Frame 1', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2[:h, :w])
    axes[0, 1].set_title('Frame 2', fontsize=11)
    axes[0, 1].axis('off')

    # magnitude flow - light yellow = heavy traffic, dark = no traffic
    im = axes[1, 0].imshow(flow_magnitude, cmap='hot', vmin=0, vmax=20)
    plt.colorbar(im, ax=axes[1, 0], label='Flow magnitude (px)')
    axes[1, 0].set_title('Flow magnitude - light = fast movement', fontsize=11)
    axes[1, 0].axis('off')

    # flow vectors in the image (sparse - every 30 pixels)
    axes[1, 1].imshow(img1[:h, :w], alpha=0.7)
    step = 30
    y_coords, x_coords = np.mgrid[step//2:h:step, step//2:w:step]
    fx = flow[0][y_coords, x_coords]
    fy = flow[1][y_coords, x_coords]

    magnitude = np.sqrt(fx**2, fy**2)
    axes[1, 1].quiver(x_coords, y_coords, fx, -fy,  # -fy because Y is inverted in the images
                      magnitude, cmap='plasma', scale=200, width=0.002, alpha=0.8)
    axes
    axes[1, 1].set_title('Flow vectors - arrows indicating the direction of movement', fontsize=11)
    axes[1, 1].axis('off')

    plt.suptitle(f'RAFT Optical Flow {cam_channel}', fontsize=13)
    plt.tight_layout()
    plt.savefig('outputs/optical_flow.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: outputs/optical_flow.png")
    print(f"Mean flow: {flow_magnitude.mean():.2f}px")
    print(f"Max flow: {flow_magnitude.max():.2f}px")

flow = run_optical_flow(scene_idx=0, cam_channel='CAM_FRONT')