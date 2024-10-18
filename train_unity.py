# -*- coding: utf-8 -*-
# TODO:
#  1. logging visuals and weights and losses
#  2. visualize loss gradients w.r.t. ray density samples and ray intensity samples
#  3. shadow difficulty hypothesis 1: losses and sampling lead to misaligned gradients
#  4. shadow difficulty hypothesis 2: model too smooth for learning high density obstacles
import torch
from torch.utils import data
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import json
from PIL import Image
from datetime import datetime
from torch.nn import functional as F

from model import SIREN, encode_position, SIREN_NeRF
from dataset_unity import Unity_Dataset

def generate_coord_normalization_function(min_x, max_x, min_y, max_y, min_z, max_z):
    # produces a function that normalizes coordinates to [-1, 1]
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    def coord_normalization_function(tensor):
        x = (tensor[:, :, :, 0] - min_x) / x_range * 2 - 1
        y = (tensor[:, :, :, 1] - min_y) / y_range * 2 - 1
        z = (tensor[:, :, :, 2] - min_z) / z_range * 2 - 1
        return torch.stack([x, y, z], dim=-1)
    return coord_normalization_function


def generate_coord_denormalization_function(min_x, max_x, min_y, max_y, min_z, max_z):
    # produces a function that denormalizes coordinates from [-1, 1] to the original range
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z
    def coord_denormalization_function(tensor):
        x = (tensor[:, :, :, 0] + 1) / 2 * x_range + min_x
        y = (tensor[:, :, :, 1] + 1) / 2 * y_range + min_y
        z = (tensor[:, :, :, 2] + 1) / 2 * z_range + min_z
        return torch.stack([x, y, z], dim=-1)
    return coord_denormalization_function


def generate_pixel_intensity_normalization_function(max_l):
    # produces a function that normalized pixel intensities to [0, 1]
    def pixel_intensity_normalization_function(l):
        l = l / max_l
        return l
    return pixel_intensity_normalization_function


def generate_pixel_intensity_denormalization_function(max_l):
    # produces a function that denormalizes pixel intensities from [0, 1] to the original range
    def pixel_intensity_denormalization_function(l):
        l = l * max_l
        return l
    return pixel_intensity_denormalization_function


def render_image_raymarch_nerf(radiance_field, pos_encode_func, dir_encode_func, ray_origins, ray_endpoints, num_bins):
    # radiance_field: [B, H, W, 3] -> [B, H, W, 2] including densities and intensities
    # encode_func: [..., 3] -> [..., enc_dim]
    # ray_origins: [B, 3]
    # ray_endpoints: [B, H, W, 3]
    # num_bins: int
    # return: [B, H, W, 1]
    B, H, W, _ = ray_endpoints.shape
    # numerical integration with quadrature
    num_rays = B * H * W
    ray_origins = ray_origins.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1).reshape(num_rays, 3) # [num_rays, 3]
    ray_endpoints = ray_endpoints.reshape(num_rays, 3) # [num_rays, 3]
    ray_displacements = ray_endpoints - ray_origins # [num_rays, 3]
    ray_lengths = torch.norm(ray_displacements, dim=-1) # [num_rays]
    ray_directions = ray_displacements / ray_lengths.unsqueeze(-1) # [num_rays, 3]
    eval_bins_bounds = torch.linspace(0.0, 1.0, num_bins + 1).cuda() # [num_bins + 1]
    eval_bins_bounds = eval_bins_bounds.unsqueeze(0).unsqueeze(-1) # [1, num_bins + 1, 1]
    eval_bins_bounds = eval_bins_bounds * ray_displacements.unsqueeze(1) # [num_rays, num_bins + 1, 3]
    eval_bins_bounds = eval_bins_bounds + ray_origins.unsqueeze(1) # [num_rays, num_bins + 1, 3]
    # [num_rays, num_bins, 3]
    eval_bins_starts = eval_bins_bounds[:, :-1, :] # [num_rays, num_bins, 3]
    eval_bins_ends = eval_bins_bounds[:, 1:, :] # [num_rays, num_bins, 3]
    # randomly sample a point within each bin
    eval_points = torch.rand([num_rays, num_bins, 1]).cuda() # [num_rays, num_bins, 1]
    eval_points = eval_points * (eval_bins_ends - eval_bins_starts) # [num_rays, num_bins, 3]
    eval_points += eval_bins_starts # [num_rays, num_bins, 3]
    # add the ray_origins and ray_endpoints to the list of samples, iterate from sample 1 instead of 0
    num_samples = num_bins + 1
    eval_points = torch.cat([ray_origins.unsqueeze(1), eval_points, ray_endpoints.unsqueeze(1)], dim=1) # [num_rays, num_samples + 1, 3]
    eval_displacements = eval_points[:, 1:, :] - eval_points[:, :-1, :] # [num_rays, num_samples, 3]
    delta = torch.norm(eval_displacements, dim=-1) # [num_rays, num_samples]
    eval_points = eval_points[:, 1:, :] # [num_rays, num_samples, 3]
    # encode samples from positions and directions
    encoded_positions = pos_encode_func(eval_points) # [num_rays, num_samples, enc_dim]
    encoded_directions = dir_encode_func(ray_directions.unsqueeze(1).repeat(1, num_samples, 1)) # [num_rays, num_samples, enc_dim]
    # evaluate radiance field
    sample_radiance_field = radiance_field(encoded_positions, encoded_directions) # [num_rays, num_samples, 2]
    sigma = sample_radiance_field[:, :, 0] # [num_rays, num_samples]
    intensity = sample_radiance_field[:, :, 1] # [num_rays, num_samples]
    T = sigma * delta # [num_rays, num_samples]
    pixel_T = torch.exp(-torch.sum(T[:, :-1], dim=-1)) # [num_ray], accumulated transmittance, the probability of ray not being absorbed before ray_endpoints
    pixel_sigma = sigma[:, -1] # [num_rays]
    pixel_intensity = intensity[:, -1] # [num_rays]
    pixel_alpha = 1.0 - torch.exp(-pixel_sigma) # [num_rays], the probability of ray being reflected at ray_endpoints
    pixels = pixel_T * pixel_alpha * pixel_intensity # [num_rays]
    pixels = pixels.reshape(B, H, W, 1) # [B, H, W, 1]
    return pixels


def calculate_bounds(dataset, batch_size):
    ### get train dataset statistics for normalization ###
    train_dataloader = data.DataLoader(dataset, batch_size=batch_size)
    max_x = 0.0
    max_y = 0.0
    max_z = 0.0
    min_x = 0.0
    min_y = 0.0
    min_z = 0.0
    max_l = 0.0
    print('calculating dataset statistics')
    for datapoints in train_dataloader:
        pixel_coords = datapoints['pixel_coords'].cuda().float() # [B, H, W, 3]
        gt = datapoints['imgs'].cuda().float()
        max_x = max(max_x, pixel_coords[:, :, :, 0].max())
        max_y = max(max_y, pixel_coords[:, :, :, 1].max())
        max_z = max(max_z, pixel_coords[:, :, :, 2].max())
        min_x = min(min_x, pixel_coords[:, :, :, 0].min())
        min_y = min(min_y, pixel_coords[:, :, :, 1].min())
        min_z = min(min_z, pixel_coords[:, :, :, 2].min())
        max_l = max(max_l, gt.max())
    print('max_x: {}, max_y: {}, max_z: {}, min_x: {}, min_y: {}, min_z: {}, max_l: {}'.format(max_x, max_y, max_z, min_x, min_y, min_z, max_l))
    return max_x, max_y, max_z, min_x, min_y, min_z, max_l


def main():
    ### training hyperparameters ###
    N_EPOCH = 10001  # training epoch for each set of images
    EVAL_INTERVAL = 5  # number of epochs between visualizations
    TRAIN_BATCH_SIZE = 2 # number of images per backpropagation during training
    TEST_BATCH_SIZE = 1 # number of images per  forward propagation during testing
    NUM_TEST_IMAGES = 4 # total number of images to test on
    SAMPLE_HEIGHT = 64
    SAMPLE_WIDTH = 64
    NUM_SAMPLES_PER_RAY = 64 # number of samples along each ray = number of bins for numerical integration + 1, used for anisotropic
    LEARNING_RATE = 2e-4
    MLP_LAYER_WIDTH = 128
    ISOTROPIC = False
    ISOTROPIC_RF = True
    POSITIONAL_ENCODING_LEVEL = 10
    POSITIONAL_ENCODING_INCLUDE_INPUT = True
    DIRECTIONAL_ENCODING_LEVEL = 4
    DIRECTIONAL_ENCODING_INCLUDE_INPUT = True
    DATA_DIR = 'C:\\Users\\alast\\Documents\\GitHub\\neural_sound_field\\example\\RubiksCubeInSphere'
    LOG_PARENT_DIR = './logs'
    EVAL_SAVE_WEIGHTS = True
    EVAL_SAVE_VISUALS = True
    EVAL_SHOW_VISUALS = True
    COORD_NORMALIZATION = False # set normalizations to false for easier debugging

    ymdhms = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOG_PARENT_DIR, ymdhms)
    os.makedirs(log_dir, exist_ok=True)
    metadata = {
        'N_EPOCH': N_EPOCH,
        'EVAL_INTERVAL': EVAL_INTERVAL,
        'TRAIN_BATCH_SIZE': TRAIN_BATCH_SIZE,
        'TEST_BATCH_SIZE': TEST_BATCH_SIZE,
        'NUM_TEST_IMAGES': NUM_TEST_IMAGES,
        'SAMPLE_HEIGHT': SAMPLE_HEIGHT,
        'SAMPLE_WIDTH': SAMPLE_WIDTH,
        'NUM_SAMPLES_PER_RAY': NUM_SAMPLES_PER_RAY,
        'LEARNING_RATE': LEARNING_RATE,
        'MLP_LAYER_WIDTH': MLP_LAYER_WIDTH,
        'ISOTROPIC': ISOTROPIC,
        'ISOTROPIC_RF': ISOTROPIC_RF,
        'POSITIONAL_ENCODING_LEVEL': POSITIONAL_ENCODING_LEVEL,
        'POSITIONAL_ENCODING_INCLUDE_INPUT': POSITIONAL_ENCODING_INCLUDE_INPUT,
        'DIRECTIONAL_ENCODING_LEVEL': DIRECTIONAL_ENCODING_LEVEL,
        'DIRECTIONAL_ENCODING_INCLUDE_INPUT': DIRECTIONAL_ENCODING_INCLUDE_INPUT,
        'DATA_DIR': DATA_DIR,
        'EVAL_SAVE_WEIGHTS': EVAL_SAVE_WEIGHTS,
        'EVAL_SAVE_VISUALS': EVAL_SAVE_VISUALS,
        'EVAL_SHOW_VISUALS': EVAL_SHOW_VISUALS,
        'COORD_NORMALIZATION': COORD_NORMALIZATION,
    }
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    num_eval_per_training_batch = TRAIN_BATCH_SIZE * SAMPLE_HEIGHT * SAMPLE_WIDTH
    num_eval_per_training_batch *= NUM_SAMPLES_PER_RAY if not ISOTROPIC else 1
    print('num_eval_per_training_batch: ', num_eval_per_training_batch)
    # isotropic num_eval per patch = TRAIN_BATCH_SIZE * SAMPLE_HEIGHT * SAMPLE_WIDTH
    # anisotropic num_eval per patch = TRAIN_BATCH_SIZE * SAMPLE_HEIGHT * SAMPLE_WIDTH * NUM_SAMPLES
    # for reference, NeRF uses batch size of 4096 rays, each with 64+128 sampling points, resulting in 786432 evaluations
    # or, approximately 1 million evaluations per batch

    ### initialize dataset - 80% train 20% test ###
    train_dataset = Unity_Dataset(DATA_DIR, partition='train', isotropic=ISOTROPIC, h=SAMPLE_HEIGHT, w=SAMPLE_WIDTH)
    test_dataset = Unity_Dataset(DATA_DIR, partition='test', isotropic=ISOTROPIC, h=SAMPLE_HEIGHT, w=SAMPLE_WIDTH)

    ### calibrate normalization functions dataset statistics ###
    max_x, max_y, max_z, min_x, min_y, min_z, max_l = calculate_bounds(train_dataset, batch_size=TEST_BATCH_SIZE)
    if COORD_NORMALIZATION:
        coord_norm_func = generate_coord_normalization_function(min_x, max_x, min_y, max_y, min_z, max_z)
    else:
        coord_norm_func = lambda X: X
    intensity_norm_func = generate_pixel_intensity_normalization_function(max_l)
    intensity_denorm_func = generate_pixel_intensity_denormalization_function(max_l)

    ### initialize neural network ###
    positional_encoding_input_dim = 3
    positional_encoding_dim = positional_encoding_input_dim * (2 * POSITIONAL_ENCODING_LEVEL + POSITIONAL_ENCODING_INCLUDE_INPUT)
    directional_encoding_input_dim = 3
    directional_encoding_dim = directional_encoding_input_dim * (2 * DIRECTIONAL_ENCODING_LEVEL + DIRECTIONAL_ENCODING_INCLUDE_INPUT)
    pos_encode_func = lambda X: encode_position(X, levels=POSITIONAL_ENCODING_LEVEL, inc_input=POSITIONAL_ENCODING_INCLUDE_INPUT)
    dir_encode_func = lambda X: encode_position(X, levels=DIRECTIONAL_ENCODING_LEVEL, inc_input=DIRECTIONAL_ENCODING_INCLUDE_INPUT)
    if ISOTROPIC:
        model = SIREN(pos_in_dims=positional_encoding_dim, D=MLP_LAYER_WIDTH).cuda()
    else:
        model = SIREN_NeRF(pos_in_dims=positional_encoding_dim, dir_in_dims=directional_encoding_dim, D=MLP_LAYER_WIDTH, isotropic_intensity=ISOTROPIC_RF).cuda()
    ### initialize optimizer ###
    if ISOTROPIC_RF:
        my_list = ['intensity_param']
        intensity_param = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
        base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))
        base_params = [kv[1] for kv in base_params]
        opt_nerf = torch.optim.Adam([{'params':base_params},
                                     {'params':intensity_param[0][1], 'lr': LEARNING_RATE * 1e-2}], lr=LEARNING_RATE)
    else:
        opt_nerf = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('training')
    ### training loop ###
    for e in range(N_EPOCH):
        ### eval loop ###
        ### visual evaluation - sample coordinates from test dataset, render them, and compare to ground truth ###
        if e % EVAL_INTERVAL == 0:
            with torch.no_grad():
                model.eval()
                test_dataloader = data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)
                num_tested = 0
                gts = []
                preds = []
                for datapoints in test_dataloader:
                    if num_tested >= NUM_TEST_IMAGES:
                        break
                    pixel_coords = datapoints['pixel_coords'].cuda().float()
                    pixel_coords = coord_norm_func(pixel_coords)
                    gt = datapoints['imgs'].float().cpu().numpy()
                    ### forward pass renders ###
                    if ISOTROPIC:
                        pos_enc = pos_encode_func(pixel_coords)
                        pred = model(pos_enc)
                    else:
                        ray_origins = datapoints['sensor_coords'].cuda().float()  # [B, 3]
                        ray_endpoints = pixel_coords  # [B, H, W, 3]
                        pred = render_image_raymarch_nerf(model, pos_encode_func, dir_encode_func, ray_origins,
                                                         ray_endpoints, NUM_SAMPLES_PER_RAY - 1)  # [B, H, W, 1]
                    pred = intensity_denorm_func(pred).squeeze(-1).detach().cpu().numpy()
                    preds.append(pred)
                    gts.append(gt)
                    num_tested += pred.shape[0]
                gts = np.concatenate(gts, axis=0)
                preds = np.concatenate(preds, axis=0)
                if EVAL_SHOW_VISUALS:
                    ### plot and show gif ###
                    fig = plt.figure(figsize=(10, 5))
                    ax_img = fig.add_subplot(121)
                    ax_atl_pred = fig.add_subplot(122)
                    for i in range(preds.shape[0]):
                        ax_atl_pred.cla()
                        ax_atl_pred.imshow(preds[i], cmap='gray')
                        ax_atl_pred.axis('off')
                        # plot plane
                        ax_img.cla()
                        ax_img.imshow(gts[i], cmap='gray')
                        ax_img.set_title(i) # type: ignore
                        ax_img.axis('off')
                        plt.pause(1.0)
                        # create gif
                        fig.canvas.draw()  # draw the canvas, cache the renderer
                    plt.close('all')
                if EVAL_SAVE_VISUALS:
                    ### save eval images ###
                    for i in range(preds.shape[0]):
                        save_path = os.path.join(log_dir, f'epoch{e:06d}', f'{i:06d}.png')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        comparison = np.concatenate([gts[i], preds[i]], axis=1)
                        Image.fromarray(comparison).convert('RGB').save(save_path)
                if EVAL_SAVE_WEIGHTS:
                    ### save weights ###
                    save_path = os.path.join(log_dir, f'epoch{e:06d}', f'weights.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
        start_epoch = time.time()
        torch.cuda.synchronize()
        model.train()
        loss_epoch = []
        train_dataloader = data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
        ### training - sample images and metadata from dataloader and train loss ###
        for datapoints in train_dataloader:
            pixel_coords = datapoints['pixel_coords'].cuda().float()
            pixel_coords = coord_norm_func(pixel_coords)
            gt = datapoints['imgs'].cuda().float()
            gt = intensity_norm_func(gt)
            if ISOTROPIC:
                ### preprocess coordinates and directions for positional encoding ###
                pos_enc = pos_encode_func(pixel_coords)
                ### forward pass ###
                pred = model(pos_enc) # [B, H, W, 1]
                loss = F.mse_loss(pred.squeeze(-1), gt)  # L2 # scalar
            else:
                ### prepare rays ###
                ray_origins = datapoints['sensor_coords'].cuda().float() # [B, 3]
                ray_endpoints = pixel_coords # [B, H, W, 3]
                ### forward pass ###
                pred = render_image_raymarch_nerf(model, pos_encode_func, dir_encode_func, ray_origins, ray_endpoints, NUM_SAMPLES_PER_RAY - 1) # [B, H, W, 1]
                loss = F.mse_loss(pred.squeeze(-1), gt)  # L2 # scalar
            ### backward pass ###
            loss.backward()
            opt_nerf.step()
            opt_nerf.zero_grad()
            loss_epoch.append(loss)
        ### print epoch statistics ###
        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        loss_epoch_mean = torch.stack(loss_epoch).mean().item()
        print(f"epoch: {e}", f"average loss: {loss_epoch_mean}", f"elapsed time: {elapsed}")


if __name__ ==  '__main__':
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    main()
            
            