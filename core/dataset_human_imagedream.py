import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import re

from core.utils import get_rays

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Imagedream_LGM_dataset(Dataset):
    def __init__(self, render_path, opt, training=True, front_input=False, all_data=False, white_bg=False):
        if white_bg == False:
            assert False, "Only white background is supported for this dataset"
            
        self.training = training
        self.front_input = front_input
        self.normalize_mean = (0.5, 0.5, 0.5)
        self.normalize_std = (0.5, 0.5, 0.5)

        self.opt = opt

        self.num_views = opt.num_views
        if self.training == False:
            self.num_views = 32
        self.num_input_views = opt.num_input_views
        self.input_size = opt.input_size
        self.output_size = opt.output_size
        self.prob_grid_distortion = opt.prob_grid_distortion
        self.prob_cam_jitter = opt.prob_cam_jitter
        
        self.all_data = all_data

        self.dataset_name = self.extract_xxx_part(render_path.split('/')[-1])

        self.blender2opengl = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
        
        self.items = []
        for subject in sorted(os.listdir(render_path)):
            self.items.append(os.path.join(render_path, subject))

        self.fovy = self.opt.fovy
        self.zfar = self.opt.zfar
        self.znear = self.opt.znear
        self.cam_radius = self.opt.cam_radius
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.zfar + self.znear) / (self.zfar - self.znear)
        self.proj_matrix[3, 2] = - (self.zfar * self.znear) / (self.zfar - self.znear)
        self.proj_matrix[2, 3] = 1
    
    def __len__(self):
        return len(self.items)
    
    @staticmethod
    def extract_xxx_part(s):
        match = re.search(r'ImagedreamLGM_(.*?)_132view', s)
        if match:
            return match.group(1)
        else:
            assert False, f"Cannot extract 'ImagedreamLGM_xxx_132view' part from {s}"
            return None

    def __getitem__(self, idx):
        assert self.num_input_views == 4, "Only 4 input views are supported for this dataset"
        
        rendering_path = self.items[idx]
        results = {}
        subject_name = rendering_path.split('/')[-1]
        
        imagedream_images = []
        imagedream_masks = []
        imagedream_cam_poses = []

        first_gt_image_idx = np.random.permutation(np.arange(0, 31))[:1]
        if self.front_input:
            first_gt_image_idx = np.random.permutation(np.arange(8, 9))[:1]
        second_gt_image_idx = (first_gt_image_idx + 8) % 32
        third_gt_image_idx = (first_gt_image_idx + 16) % 32
        fourth_gt_image_idx = (first_gt_image_idx + 24) % 32
        imagedream_image_idx = np.concatenate([first_gt_image_idx+100, second_gt_image_idx+100, third_gt_image_idx+100, fourth_gt_image_idx+100])
        
        for imagedream_idx in imagedream_image_idx:
            imagedream_image_path = os.path.join(rendering_path, 'rgb_132_{}.png'.format(imagedream_idx))
            imagedream_image = np.array(Image.open(imagedream_image_path), np.uint8)
            imagedream_image = torch.from_numpy(imagedream_image.astype(np.float32) / 255)

            imagedream_c2w_path = os.path.join(rendering_path, '132_{}_RT.txt'.format(imagedream_idx))
            imagedream_c2w_pose = torch.tensor(np.loadtxt(imagedream_c2w_path)).float().reshape(4, 4)

            imagedream_image = imagedream_image.permute(2, 0, 1)
            imagedream_mask = imagedream_image[3:4]
            imagedream_image = imagedream_image[:3] * imagedream_mask + (1 - imagedream_mask)

            imagedream_images.append(imagedream_image)
            imagedream_masks.append(imagedream_mask.squeeze(0))
            imagedream_cam_poses.append(imagedream_c2w_pose)

        imagedream_images = torch.stack(imagedream_images, dim=0)
        imagedream_masks = torch.stack(imagedream_masks, dim=0)
        imagedream_cam_poses = torch.stack(imagedream_cam_poses, dim=0)

        imagedream_images = F.interpolate(imagedream_images.clone(), size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        imagedream_images = TF.normalize(imagedream_images, self.normalize_mean, self.normalize_std)

        results['imagedream_images_gt'] = imagedream_images

        imagedream_cam_poses_opengl = self.blender2opengl.unsqueeze(0)@imagedream_cam_poses

        imagedream_translation = imagedream_cam_poses[:, :3, 3]
        imagedream_translation = imagedream_translation / imagedream_translation.norm(dim=-1, keepdim=True)
        imagedream_cam_poses[:, :3, 3] = imagedream_translation

        results['imagedream_cam_poses_gt'] = imagedream_cam_poses

        vid_cnt = 0

        lgm_images = []
        lgm_normals = []
        lgm_masks = []
        lgm_cam_poses = []

        vids = np.random.permutation(132).tolist()

        if self.training == False:
            vids = np.arange(100, 132).tolist()

        for vid in vids: 
            lgm_image_path = os.path.join(rendering_path, 'rgb_132_{}.png'.format(vid))
            lgm_c2w_path = os.path.join(rendering_path, '132_{}_RT.txt'.format(vid))
            lgm_normal_path = os.path.join(rendering_path, 'normals_132_{}.png'.format(vid))

            lgm_image = np.array(Image.open(lgm_image_path), np.uint8)
            lgm_image = torch.from_numpy(lgm_image.astype(np.float32) / 255)

            lgm_c2w = torch.tensor(np.loadtxt(lgm_c2w_path)).float().reshape(4, 4)
            
            lgm_image = lgm_image.permute(2, 0, 1)
            lgm_mask = lgm_image[3:4]
            lgm_image = lgm_image[:3] * lgm_mask + (1 - lgm_mask)

            lgm_normal = np.array(Image.open(lgm_normal_path), np.uint8)
            lgm_normal = torch.from_numpy(lgm_normal.astype(np.float32) / 255)
            lgm_normal = lgm_normal.permute(2, 0, 1)
            lgm_normal = torch.nn.functional.normalize(lgm_normal, p=2, dim=0)
            lgm_normal = lgm_normal * lgm_mask + (1 - lgm_mask) * 0.5
            lgm_normal = lgm_normal * 2 - 1

            lgm_images.append(lgm_image)
            lgm_normals.append(lgm_normal)
            lgm_masks.append(lgm_mask.squeeze(0))
            lgm_cam_poses.append(lgm_c2w)

            vid_cnt += 1

            if vid_cnt == self.num_views and self.training == True:
                break

        lgm_images = torch.stack(lgm_images, dim=0)
        lgm_normals = torch.stack(lgm_normals, dim=0)
        lgm_masks = torch.stack(lgm_masks, dim=0)
        lgm_cam_poses = torch.stack(lgm_cam_poses, dim=0)

        results['images_output'] = F.interpolate(lgm_images, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        results['normals_output'] = F.interpolate(lgm_normals, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        results['masks_output'] = F.interpolate(lgm_masks.unsqueeze(1), size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)

        lgm_cam_poses_opengl = self.blender2opengl.unsqueeze(0)@lgm_cam_poses

        lgm_all_cam_poses_opengl = torch.cat([imagedream_cam_poses_opengl, lgm_cam_poses_opengl], dim=0)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(lgm_all_cam_poses_opengl[0])
        lgm_all_cam_poses = transform.unsqueeze(0) @ lgm_all_cam_poses_opengl

        lgm_cam_poses_input = lgm_all_cam_poses[:self.num_input_views].clone()
        rays_embeddings = []
        for i in range(self.num_input_views):
            rays_o, rays_d = get_rays(lgm_cam_poses_input[i], self.input_size, self.input_size, self.fovy)
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)
            rays_embeddings.append(rays_plucker)
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()

        results['lgm_cam_poses_input_embedding'] = rays_embeddings

        context_img_idx = np.random.permutation(np.arange(101, 114))[0].tolist()
        context_image_path = os.path.join(rendering_path, 'rgb_132_{}.png'.format(context_img_idx))
        if self.front_input:
            prompt_img_idx = np.random.permutation(np.arange(108, 109))[0].tolist()
            context_image_path = os.path.join(rendering_path, 'rgb_132_{}.png'.format(prompt_img_idx))
        context_image = np.array(Image.open(context_image_path), np.uint8)
        context_image = torch.from_numpy(context_image.astype(np.float32) / 255)
        context_image = context_image.permute(2, 0, 1)
        context_mask = context_image[3:4]
        context_image = context_image[:3] * context_mask + (1 - context_mask)

        context_image_input = F.interpolate(context_image.unsqueeze(0), size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        context_image_input = TF.normalize(context_image_input, self.normalize_mean, self.normalize_std)
        context_cam_pos = torch.zeros_like(lgm_cam_poses[0])
        context_ray_o, context_ray_d = get_rays(context_cam_pos, self.input_size, self.input_size, self.fovy)
        context_ray_embedding = torch.cat([torch.cross(context_ray_o, context_ray_d, dim=-1), context_ray_d], dim=-1).unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        results['context_image'] = context_image_input
        results['context_ray_embedding'] = context_ray_embedding
        
        lgm_all_cam_poses[:, :3, 1:3] *= -1

        cam_view = torch.inverse(lgm_all_cam_poses).transpose(1, 2)
        cam_view_proj = cam_view @ self.proj_matrix
        cam_pos = - lgm_all_cam_poses[:, :3, 3]

        results['cam_view_imagedream'] = cam_view[:self.num_input_views]
        results['cam_view_proj_imagedream'] = cam_view_proj[:self.num_input_views]
        results['cam_pos_imagedream'] = cam_pos[:self.num_input_views]

        results['cam_view'] = cam_view[self.num_input_views:]
        results['cam_view_proj'] = cam_view_proj[self.num_input_views:]
        results['cam_pos'] = cam_pos[self.num_input_views:]

        results['dataset'] = self.dataset_name
        results['subject_name'] = subject_name

        return results
