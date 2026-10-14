from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)

sample = nusc.sample[0]

cam_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, channel in zip(axes.flatten(), cam_channels):
    cam_token = sample['data'][channel]
    cam_data = nusc.get('sample_data', cam_token)
    img = plt.imread(f"./data/nuscenes/{cam_data['filename']}")
    ax.imshow(img)
    ax.set_title(channel, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/six_cameras.png', dpi=150)
plt.show()