"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import json
import glob
import mrcfile
import plyfile
import skimage.measure
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils

from core.camera_utils import LookAtPoseSampler, FOV_to_intrinsics, batchify_cam_params


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(flag, network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    if flag:
        print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def generator_pose_conditioning(c, gpc_reg_prob):
    # alpha = min(1 / (gpc_reg_fade_kimg * 1e3), 1) if gpc_reg_fade_kimg > 0 else 1
    alpha = 1
    swapping_prob = (1 - alpha) * 1 + alpha * gpc_reg_prob if gpc_reg_prob is not None else None
    if swapping_prob is not None:
        c_swapped = torch.roll(c.clone(), 1, 0)
        c_cond = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
    else:
        c_cond = torch.zeros_like(c, device=c.device)
    return c_cond


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, c_src, x_ref, y_ref, c_ref, filename):
    N, C, H, W = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, c_ref, masks=masks)
    x_fake_rgb, x_fake_depth = x_fake['rgb_image'], x_fake['depth_image']
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake_rgb) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake_rgb, s_src, c_src, masks=masks)
    x_rec_rgb, x_rec_depth = x_rec['rgb_image'], x_rec['depth_image']
    x_concat = [x_src, x_ref, x_fake_rgb, x_rec_rgb]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, c_ref, y_trg_list, z_trg_list, psi, filename):
    device = x_src.device
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    
    c2w = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intr = FOV_to_intrinsics(device=device)
    c =  torch.cat([c2w.reshape(-1, 16), intr.reshape(-1, 9)], 1)

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(device)
        c_many = c.repeat(10000, 1)
        y_many = torch.LongTensor(10000).to(device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, c_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            # style truncatation
            s_trg = nets.mapping_network(z_trg, c_ref, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, c_ref, masks=masks)
            x_fake_rgb, x_fake_depth = x_fake['rgb_image'], x_fake['depth_image']
            x_concat += [x_fake_rgb]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, c_ref, y_ref, filename):
    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref) 
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)

    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, c_ref, masks=masks)
        x_fake_rgb, x_fake_depth = x_fake['rgb_image'], x_fake['depth_image']
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake_rgb], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N+1, filename)
    del x_concat


@torch.no_grad()
def pivot_image(nets, args, x_src, y_src, filename):
    device = x_src.device
    N, C, H, W = x_src.size()

    z = torch.randn(N, args.latent_dim).to(device)
    c = batchify_cam_params(device, N)
    s = nets.mapping_network(z, c, y_src)

    c_list = c.unsqueeze(1).repeat(1, N, 1)
    
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_concat = [x_src]
    for i, c in enumerate(c_list):
        x_fake = nets.generator(x_src, s, c, masks=masks)
        x_fake_rgb, x_fake_depth = x_fake['rgb_image'], x_fake['depth_image']
        x_concat += [x_fake_rgb]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat



@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src, c_src = inputs.x_src, inputs.y_src, inputs.c_src
    x_ref, y_ref, c_ref = inputs.x_ref, inputs.y_ref, inputs.c_ref

    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, c_src, x_ref, y_ref, c_ref, filename)

    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, c_ref, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, c_ref, y_ref, filename)

    # pivot image synthesis
    filename = ospj(args.sample_dir, '%06d_pivot.jpg' % (step))
    pivot_image(nets, args, x_src, y_src, filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255


# ======================= #
# shape-related functions #
# ======================= #

def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # try:
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )
    # except:
    #     pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the out of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
    print(f"wrote to {ply_filename_out}")


def convert_mrc(input_filename, output_filename, isosurface_level=1):
    with mrcfile.open(input_filename) as mrc:
        convert_sdf_samples_to_ply(np.transpose(mrc.data, (2, 1, 0)), [0, 0, 0], 1, output_filename, level=isosurface_level)


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


@torch.no_grad()
def shape_latent(nets, args, x_src, x_ref, y_ref, fname, guided='latent'):
    device = x_src.device
    max_batch=1000000
    N = x_src.size(0)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    c2w = LookAtPoseSampler.sample(np.pi/2, np.pi/2, torch.tensor([0, 0, 0.2]), radius=2.7)
    intr = FOV_to_intrinsics()
    c =  torch.cat([c2w.reshape(-1, 16), intr.reshape(-1, 9)], 1)

    # style truncatation
    s_trg = nets.style_encoder(x_ref, y_ref)

    samples, voxel_origin, voxel_size = create_samples(N=128, voxel_origin=[0, 0, 0], cube_length=args.box_warp * 1)#.reshape(1, -1, 3)
    samples = samples.repeat(N, 1, 1).to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3),  device=device).repeat(N, 1, 1)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        while head < samples.shape[1]:
            depths = nets.generator.sample(samples[:, head:head+max_batch],
                                            transformed_ray_directions_expanded[:, :samples.shape[1]-head],
                                            x_src, s_trg, masks)['sigma']
            sigmas[:, head:head+max_batch] = depths
            head += max_batch
            pbar.update(max_batch)

    sigmas = sigmas.reshape((N, 128, 128, 128)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # # Trim the border of the extracted cube
    # pad = int(30 * 128 / 64)
    # pad_value = -1000
    # sigmas[:pad] = pad_value
    # sigmas[-pad:] = pad_value
    # sigmas[:, :pad] = pad_value
    # sigmas[:, -pad:] = pad_value
    # sigmas[:, :, :pad] = pad_value
    # sigmas[:, :, -pad:] = pad_value
    
    # # make files
    # convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, fname, level=10)
    with mrcfile.new_mmap(fname, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        mrc.data[:] = sigmas
