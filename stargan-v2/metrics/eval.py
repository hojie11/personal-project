"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

from metrics.fid import calculate_fid_given_paths
from core.data_loader import get_eval_loader
from core import utils


@torch.no_grad()
def calculate_metrics(nets, args, step, mode):
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = torch.device(args.gpu)

    domains = os.listdir(args.val_img_dir)
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]

        if mode == 'reference':
            path_ref = os.path.join(args.val_img_dir, trg_domain, 'imgFiles')
            loader_ref = get_eval_loader(args,
                                         root=path_ref,
                                         json_pth=args.cameras,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False,
                                         drop_last=True)

        for src_idx, src_domain in enumerate(src_domains):
            path_src = os.path.join(args.val_img_dir, src_domain, 'imgFiles')
            loader_src = get_eval_loader(args,
                                         root=path_src,
                                         json_pth=args.cameras,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False)

            task = '%s2%s' % (src_domain, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)

            for i, x in enumerate(tqdm(loader_src, total=len(loader_src))):
                x_src, c_src = x
                x_src, c_src = x_src.to(device), c_src.to(device)
                N = x_src.size(0)
                y_trg = torch.tensor([trg_idx] * N).to(device)
                masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                # generate 10 outputs from the same input
                group_of_images = []
                for j in range(args.num_outs_per_domain):
                    if mode == 'latent':
                        z_trg = torch.randn(N, args.latent_dim).to(device)
                        s_trg = nets.mapping_network(z_trg, c_src, y_trg)
                        x_fake = nets.generator(x_src, s_trg, c_src, masks=masks)
                    else:
                        try:
                            x_ref = next(iter_ref).to(device)
                        except:
                            iter_ref = iter(loader_ref)
                            x_ref, c_ref = next(iter_ref)
                            x_ref, c_ref = x_ref.to(device), c_ref.to(device)

                        if x_ref.size(0) > N:
                            x_ref = x_ref[:N]
                        s_trg = nets.style_encoder(x_ref, y_trg)
                        x_fake = nets.generator(x_src, s_trg, c_ref, masks=masks)

                    x_fake_rgb, x_fake_depth = x_fake['rgb_image'], x_fake['depth_image']
                    group_of_images.append(x_fake_rgb)

                    # save generated images to calculate FID later
                    for k in range(N):
                        filename = os.path.join(
                            path_fake,
                            '%.4i_%.2i.png' % (i*args.val_batch_size+(k+1), j+1))
                        utils.save_image(x_fake_rgb[k], ncol=1, filename=filename)

        # delete dataloaders
        del loader_src
        if mode == 'reference':
            del loader_ref
            del iter_ref

    # calculate and report fid values
    calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)


def calculate_fid_for_all_tasks(args, domains, step, mode):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]

        for src_domain in src_domains:
            task = '%s2%s' % (src_domain, trg_domain)
            path_real = os.path.join(args.train_img_dir, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            print('Calculating FID for %s...' % task)
            fid_value = calculate_fid_given_paths(
                args,
                paths=[path_real, path_fake],
                img_size=args.img_size,
                batch_size=args.val_batch_size)
            fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    utils.save_json(fid_values, filename)
