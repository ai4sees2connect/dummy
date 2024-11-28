ure Matching with Graph Neural Networks.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Barroso-Laguna, A., Munukutla, S., Prisacariu, V., & Brachmann, E. (2024).
Matching 2D Images in 3D: Metric Relative Pose from Metric Correspondences.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).


sudo alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc-13 100
sudo alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++-13 100

sudo alternatives --config gcc
sudo alternatives --config g++

gcc --version




The issue in the image suggests that the repository used for installing "Development Tools" is either misconfigured, outdated, or inaccessible. Here's how you can troubleshoot and resolve it:

---

### **Steps to Fix the Issue**

#### 1. **Clear Cached Metadata**
   Clean the `yum` metadata to ensure no corrupted or old files are causing the issue:
   ```bash
   sudo yum clean all
   sudo rm -rf /var/cache/yum
   ```

#### 2. **Update yum**
   Update the `yum` package manager to its latest version:
   ```bash
   sudo yum update -y
   ```

#### 3. **Enable Required Repositories**
   - List all repositories:
     ```bash
     sudo yum repolist all
     ```
   - Check if any required repository (like `amzn2-core`, `amzn2extra`) is **disabled**. If yes, enable it:
     ```bash
     sudo yum-config-manager --enable <repo_id>
     ```
     Example:
     ```bash
     sudo yum-config-manager --enable amzn2-core
     sudo yum-config-manager --enable amzn2extra-docker
     ```

#### 4. **Enable Extras Repository**
   Amazon Linux often requires enabling the `extras` repository to install certain packages:
   ```bash
   sudo amazon-linux-extras enable epel
   sudo amazon-linux-extras enable python3.8
   ```

#### 5. **Install Development Tools**
   Once the repositories are fixed, try reinstalling the "Development Tools" group:
   ```bash
   sudo yum groupinstall "Development Tools" -y
   ```

#### 6. **Check Repository URLs**
   If you still face issues, check the `.repo` files in `/etc/yum.repos.d/` to verify their `baseurl`:
   ```bash
   cat /etc/yum.repos.d/amzn2-core.repo
   ```
   Ensure the `baseurl` or `mirrorlist` points to a valid Amazon Linux repository.

---

### **If the Issue Persists**
#### Disable Problematic Repositories
   Temporarily disable specific repositories causing conflicts (e.g., `docker-ce-nightly` in your case):
   ```bash
   sudo yum-config-manager --disable docker-ce-nightly
   ```

#### Enable Missing Dependencies
   If the group install fails due to missing dependencies, try enabling optional dependencies:
   ```bash
   sudo yum install gcc gcc-c++ make automake -y
   ```

---

### **Alternative Solution: Use Amazon Extras for GCC**
If you're unable to install the whole group, you can manually install essential tools using Amazon Linux Extras:
```bash
sudo amazon-linux-extras enable epel
sudo yum install gcc gcc-c++ make -y
```

Let me know if any specific error messages appear!

#!/bin/bash

# Get the list of all repositories (repo name and status) from yum
repo_list=$(sudo yum repolist all)

# Loop through each line of the repo list
while IFS= read -r line; do
    # Extract the repo name and status from the line
    repo_name=$(echo "$line" | awk '{print $1}')
    status=$(echo "$line" | awk '{print $2}')

    # Check if the status is 'disabled'
    if [[ "$status" == "disabled" ]]; then
        echo "Enabling repository: $repo_name"
        
        # Enable the repository by changing the 'enabled' field in its repo file
        sudo sed -i "s/^enabled=0/enabled=1/" "/etc/yum.repos.d/$repo_name.repo"
        
        echo "Repository $repo_name has been enabled."
    fi
done <<< "$repo_list"

# Update yum repositories
sudo yum repolist enabled


import bpy
import math
import os
import numpy as np

def add_constraint(camera: bpy.types.Object, obj_to_track: bpy.types.Object):
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = obj_to_track
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

def set_camera_prop(camera: bpy.types.Object, focus_obj: bpy.types.Object, focal_length, name="camera_"):
    camera.name = name
    camera.data.lens = focal_length
    camera.data.dof.focus_object = focus_obj

def distribute_points_on_sphere(N_lat, N_lon):  # P (num_points) = N_lat * N_lon
    points = []
    latitudes = np.linspace(-math.pi / 2, math.pi / 2, N_lat)  # Latitude range: -π/2 to π/2
    for phi in latitudes:
        longitudes = np.linspace(0, 2 * math.pi, N_lon)
        for theta in longitudes:
            x = 50 * np.cos(phi) * np.cos(theta)
            y = 50 * np.cos(phi) * np.sin(theta)
            z = 50 * np.sin(phi)
            points.append([x, y, z])
    return np.array(points)

# Load the .blend file
blend_file_path = "mercedes.blend"  # Change this to the actual path
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

scene = bpy.context.scene
focus_object = bpy.data.objects["Mercedes"]  # Replace "Cube" with the object you want to focus on, e.g., "Mercedes"

view_layer = bpy.context.view_layer

# Add a new light source
light_data = bpy.data.lights.new(name="New Light", type='SUN')
light_data.energy = 1000
light_object = bpy.data.objects.new(name="New Light", object_data=light_data)
view_layer.active_layer_collection.collection.objects.link(light_object)
light_object.location = (4.23433, 3.2186, 2.0674)
light_object.select_set(True)
view_layer.objects.active = light_object

# Generate and position cameras
num_cameras = 110
focal_length = 100
output_dir = "./outputs2/"  # Make sure this directory exists
os.makedirs(output_dir, exist_ok=True)

points = distribute_points_on_sphere(11, 10)

for i, p in enumerate(points):
    bpy.ops.object.camera_add(location=p)
    current_camera = bpy.context.object
    set_camera_prop(current_camera, focus_object, focal_length=focal_length, name=f"cam_({i})")
    add_constraint(current_camera, focus_object)

# Render images from each camera's perspective
for obj in scene.objects:
    if obj.type == "CAMERA":
        scene.camera = obj
        output_path = os.path.join(output_dir, f"{obj.name}.png")
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered and saved: {output_path}")


# -*- coding: utf-8 -*-
"""mast3r.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1u52YMgysShrmfB6WBQZ02VwXwEhGWoe_
"""

cd/content/drive/MyDrive

!git clone --recursive https://github.com/naver/mast3r

# if you have already cloned mast3r:
# git submodule update --init --recursive

cd/content/drive/MyDrive/mast3r

!pip install -r requirements.txt
!pip install -r dust3r/requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add required packages for visloc.py
!pip install -r dust3r/requirements_optional.txt

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    images = load_images(['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png'], size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # visualize a few matches
    import numpy as np
    import torch
    import torchvision.transforms.functional
    from matplotlib import pyplot as pl

    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)

# Commented out IPython magic to ensure Python compatibility.
import os
os.makedirs('/content/drive/MyDrive/co3d_data', exist_ok=True)
# %cd /content/drive/MyDrive/co3d_data

# Step 1: Create necessary directories
!mkdir -p data/co3d_subset
# %cd data/co3d_subset

# Step 2: Clone the CO3D repository
!git clone https://github.com/facebookresearch/co3d

# Step 3: Change directory to the cloned repository
# %cd co3d

# Step 4: Download the dataset subset
!python3 ./co3d/download_dataset.py --download_folder ../ --single_sequence_subset

# Step 5: Remove unnecessary ZIP files
!rm ../*.zip

# Step 6: Navigate back to the base directory
# %cd ../../../

cd/content/drive/MyDrive/mast3r

!python3 /content/drive/MyDrive/mast3r/dust3r/datasets_preprocess/preprocess_co3d.py --co3d_dir /content/drive/MyDrive/co3d_data/data/co3d_subset --output_dir /content/drive/MyDrive/mast3r/data/co3d_subset_processed  --single_sequence_subset

# download the pretrained dust3r checkpoint



# Create directory in Google Drive
os.makedirs('/checkpoints', exist_ok=True)

# Download file into Google Drive
!wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P /content/drive/MyDrive/checkpoints/

# Step 1: Set up the environment
import os
os.makedirs('checkpoints/mast3r_demo', exist_ok=True)

# Step 2: Install dependencies (if not already installed)
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Adjust version based on your needs

# Step 3: Run the training script using the `torchrun` command
!torchrun --nproc_per_node=1 train.py \
    --train_dataset "1000 @ Co3d(split='train', ROOT='/content/drive/MyDrive/mast3r/data/co3d_subset_processed', aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', mask_bg='rand', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], n_corres=8192, nneg=0.5, transform=ColorJitter)" \
    --test_dataset "100 @ Co3d(split='test', ROOT='/content/drive/MyDrive/mast3r/data/co3d_subset_processed', resolution=(512,384), n_corres=1024, seed=777)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)" \
    --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
    --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
    --pretrained "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" \
    --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 4 --accum_iter 4 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --disable_cudnn_benchmark \
    --output_dir "checkpoints/mast3r_demo"
