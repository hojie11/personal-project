"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    args.distributed = args.world_size > 1 or args.is_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.rank = 0

    if args.is_distributed:
        args.world_size = ngpus_per_node * args.world_size
        args.num_workers = ngpus_per_node * args.num_workers
        args.batch_size = ngpus_per_node * args.batch_size
        args.val_batch_size = ngpus_per_node * args.val_batch_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.gpu = 1
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.is_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', 
                                init_method=f'tcp://127.0.0.1:{args.port}', #127.0.0.1:23456
                                world_size=args.world_size, 
                                rank=args.rank)

    args.flag = True if args.is_distributed == False or args.gpu == 0 else False
    if args.flag: print(args)
    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        assert args.cameras != None
        loaders = Munch(src=get_train_loader(args,
                                             root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=int(args.batch_size / args.world_size),
                                             prob=args.randcrop_prob,
                                             num_workers=int(args.num_workers / args.world_size)),
                        ref=get_train_loader(args,
                                             root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=int(args.batch_size / args.world_size),
                                             prob=args.randcrop_prob,
                                             num_workers=int(args.num_workers / args.world_size)),
                        val=get_test_loader(args,
                                            root=args.val_img_dir,
                                            json_pth=args.cameras,
                                            img_size=args.img_size,
                                            batch_size=int(args.val_batch_size / args.world_size),
                                            shuffle=True,
                                            num_workers=int(args.num_workers / args.world_size)))
        solver.train(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(args,
                                            root=args.src_dir,
                                            json_pth=None,
                                            img_size=args.img_size,
                                            batch_size=int(args.val_batch_size / args.world_size),
                                            shuffle=False,
                                            num_workers=int(args.num_workers / args.world_size)),
                        ref=get_test_loader(args,
                                            root=args.ref_dir,
                                            json_pth=None,
                                            img_size=args.img_size,
                                            batch_size=int(args.val_batch_size / args.world_size),
                                            shuffle=False,
                                            num_workers=int(args.num_workers / args.world_size)))
        solver.sample(loaders, shape=True)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_distributed', type=bool, default=False)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--port', type=int, default=88)
    parser.add_argument('--use_cyc_loss', type=bool, default=False)

    # model arguments
    parser.add_argument('--img_size', type=int, default=128,
                        help='Generator input image resolution')
    parser.add_argument('--discriminator_img_size', type=int, default=128,
                        help='Discriminator input image resolution')
    parser.add_argument('--generator_output_dim', type=int, default=48,
                        help='Generator output feature dimension')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')
    parser.add_argument('--pose_dim', type=int, default=25,
                        help='camera parameter dimension')
    parser.add_argument('--mlp_dim', type=int, default=32,
                        help='MLP decoder dimension')
    parser.add_argument('--decoder_input_dim', type=int, default=16,
                        help='Neural plane decoder input dimension')
    parser.add_argument('--decoder_output_dim', type=int, default=3,
                        help='Neural plane decoder input dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_pose', type=float, default=1,
                        help='Weight for pose diversity sensitive loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=10000000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=10000000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=150000,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4, #1e-4
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=6,
                        help='Number of generated images per domain during sampling')

    # rendering arguments
    parser.add_argument('--disparity_space_sampling', type=bool, default=False)
    parser.add_argument('--clamp_mode', type=str, default='softplus')
    parser.add_argument('--depth_resolution', type=int, default=48,
                        help='number of uniform samples to take per ray')
    parser.add_argument('--depth_resolution_importance', type=int, default=48,
                        help='number of importance samples to take per ray')
    parser.add_argument('--ray_start', type=float, default=2.25,
                        help='near point along each ray to start taking samples')
    parser.add_argument('--ray_end', type=float, default=3.3,
                        help='far point along each ray to stop taking samples')
    parser.add_argument('--box_warp', type=int, default=1,
                        help='the side-length of the bounding box spanned by the tri-planes\
                              box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5]')
    parser.add_argument('--avg_camera_radius', type=float, default=2.7,
                        help='used only in the visualizer to specify camera orbit radius')
    parser.add_argument('--avg_camera_pivot', type=list, default=[0, 0, 0.2],
                        help='used only in the visualizer to control center of camera rotation')
    parser.add_argument('--truncation_psi', type=float, default=0,
                        help='truncate w feature vector. default is 0, which is generate average image. \
                              the higher psi is, the more extream image is generated.')
    parser.add_argument('--truncation_cutoff', type=int, default=14,
                        help='length of truncated w vector')
    ## vv adapted from eg3d vv
    parser.add_argument('--gen_pose_cond', type=bool, default=False,
                        help='If true, enable generator pose conditioning.')
    parser.add_argument('--gpc_reg_prob', type=float, default=0.5,
                        help='strength of swapping regularization. None means no generator pose conditioning ')
    parser.add_argument('--gpc_reg_fade_kimg', type=int, default=1000,
                        help='length of swapping prob fade')

    # misc
    parser.add_argument('--mode', type=str, default='train', #required=True,
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='ffhq-256x256/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='ffhq-256x256/val',
                        help='Directory containing validation images')
    parser.add_argument('--cameras', type=str, default='ffhq-256x256/cameras.json',
                        help='File path of cameras.json')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/assets/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='neural_plane/assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='neural_plane/assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='neural_plane/assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='neural_plane/assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='neural_plane/expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='neural_plane/expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=50000)

    args = parser.parse_args()
    main(args)

"""
how to train the model?
==> python neural_plane/main.py --is_distributed=True --mode=train --batch_size=8 --gen_pose_cond=True --val_batch_size=16 --resume_iter=930000 --lambda_ds=0.5200

how to sample?
==> python neural_plane/main.py --mode=sample --val_batch_size=8 --resume_iter=920000
"""