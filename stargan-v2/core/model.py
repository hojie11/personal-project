"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import time
import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wing import FAN
from .renderer import RaySampler, ImportranceRenderer


## vv adapted from eg3d implementation vv
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, max_conv_dim=256, op=None):
        super().__init__()
        dim_in = 2**14 // op['img_size']
        self.img_size = op['img_size']
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))

        self.to_nerual_plane = nn.Parameter(torch.randn((op['img_size'], op['img_size'])))

        # down/up-sampling blocks
        repeat_num = int(np.log2(op['img_size'])) - 4
        if op['w_hpf'] > 0:
            repeat_num += 1
        for idx in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            if idx == 0:
                self.decode.insert(
                    0, AdainResBlk(dim_out, op['output_feature_dim'], op['style_dim'],
                                w_hpf=op['w_hpf'], upsample=True))  # stack-like
            else:
                self.decode.insert(
                    0, AdainResBlk(dim_out, dim_in, op['style_dim'],
                                w_hpf=op['w_hpf'], upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, op['style_dim'], w_hpf=op['w_hpf']))

        if op['w_hpf'] > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(op['w_hpf'], device)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64]):
                cache[x.size(2)] = x
            x = block(x)
        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        
        # x = torch.matmul(x, self.to_nerual_plane)
        # return self.to_rgb(x)
        return torch.matmul(x, self.to_nerual_plane)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=16, s_dim=64, c_dim=25, w_dim=512,
                 embed_features=None, num_domains=2):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0

        layers = []
        layers += [nn.Linear(z_dim + embed_features, 512)]
        layers += [nn.ReLU()]
        for _ in range(2):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, s_dim))]
        self.embed = nn.Linear(c_dim, 512)

    def forward(self, z, c, y):
        # embed, normalize, and concat inputs
        x = None
        if self.z_dim > 0:
            x = z
        if self.c_dim > 0:
            c = self.embed(c)
            x = torch.concat((x, c), dim=1) if x is not None else c

        h = self.shared(x)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, s_dim=64,
                 num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(512, s_dim)]

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, s_dim=64, c_dim=25, num_domains=2, max_conv_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.s_dim = s_dim

        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        self.main = nn.Sequential(*blocks)

        if self.c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0)

        domain_layer = []
        domain_layer += [nn.LeakyReLU(0.2)]
        domain_layer += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        domain_layer += [nn.LeakyReLU(0.2)]
        domain_layer += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.domain = nn.Sequential(*domain_layer)

    def forward(self, x, c, y):
        out = x
        for layer in self.main:
            N, C, W, H = out.shape
            if W*H == self.s_dim and self.c_dim > 0 and c is not None:
                c = self.mapping(None, c, y).unsqueeze(1)
                out = out.reshape(N, C, W*H)
                out = (out * c) * (1 / np.sqrt(self.c_dim))
                out = out.reshape(out.shape[0], out.shape[1], W, H)
                out = layer(out)
            else:
                out = layer(out)

        domain = self.domain(out)
        domain = domain.view(domain.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        domain = domain[idx, y]  # (batch)
        return domain


class Decoder(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(nn.Linear(op['input_dim'], self.hidden_dim),
                                       nn.Softplus(),
                                       nn.Linear(self.hidden_dim, 1 + op['output_dim']))
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


class NeuGenerator(nn.Module):
    def __init__(self, generator_op={}, decoder_op={}, rendering_op={}):
        super().__init__()
        self.resolution = 128
        self.backbone = Generator(op=generator_op)
        self.decoder = Decoder(op=decoder_op)
        self.ray_sampler = RaySampler()
        self.renderer = ImportranceRenderer()
        self.rendering_op = rendering_op

    def synthesis(self, x, s, c, masks=None):
        c2w = c[:, :16].reshape(-1, 4, 4)
        intr = c[:, 16:].reshape(-1, 3, 3)

        # sample a batch of rays for volume rendering
        # start = time.time()
        rays_o, rays_d = self.ray_sampler(c2w, intr, self.resolution)
        B, N_rays, _ = rays_o.shape
        # print(time.time() - start)

        # generate neural planes features
        features = self.backbone(x, s, masks) # B, C, H, W
        features = features.view(len(features), 3, 16, features.shape[-2], features.shape[-1]) # B, xyz-planes, C, H, W

        # start = time.time()
        feature_samples, depth_samples, weights_samples = self.renderer(features, self.decoder, rays_o, rays_d, self.rendering_op)
        # print(time.time() - start)
        
        # reshape renderer results
        H = W = self.resolution
        img = feature_samples.permute(0, 2, 1).reshape(B, feature_samples.shape[-1], H, W).contiguous()
        depth = depth_samples.permute(0, 2, 1).reshape(B, depth_samples.shape[-1], H, W)

        return {'rgb_image' : img[:, :3], 'depth_image' : depth}
    
    def sample(self, coordinates, directions, x, s, masks):
        features = self.backbone(x, s, masks)
        features = features.view(len(features), 3, 16, features.shape[-2], features.shape[-1]) # B, xyz-planes, C, H, W
        return self.renderer.run_model(features, self.decoder, coordinates, directions, self.rendering_op)

    def forward(self, x, s, c, masks=None):
        return self.synthesis(x, s, c, masks)


def build_model(args):
    # generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    generator = NeuGenerator(generator_op={'img_size' : args.img_size,
                                           'style_dim' : args.style_dim,
                                           'output_feature_dim' : args.generator_output_dim,
                                           'w_hpf' : args.w_hpf},
                             decoder_op={'input_dim' : args.decoder_input_dim,
                                         'output_dim' : args.decoder_output_dim},
                             rendering_op={'N_samples' : args.depth_resolution,
                                           'N_importances' : args.depth_resolution_importance,
                                           'ray_start' : args.ray_start,
                                           'ray_end' : args.ray_end,
                                           'box_warp' : args.box_warp,
                                           'is_disparity' : args.disparity_space_sampling,
                                           'clamp_mode' : args.clamp_mode})
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.pose_dim, num_domains=args.num_domains)
    style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains)
    discriminator = Discriminator(args.discriminator_img_size, args.style_dim, args.pose_dim, args.num_domains)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    if args.w_hpf > 0:
        fan = FAN(fname_pretrained=args.wing_path).eval()
        nets.fan = fan
        nets_ema.fan = fan

    return nets, nets_ema