import torch
import torch.nn as nn
import torch.nn.functional as F


class RaySampler(nn.Module):
    """
    The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
    Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, c2w, intr, resolution):
        """
        Create batches of rays and return origins and directions.

        c2w: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        rays_o: (N, M, 3)
        rays_d: (N, M, 2)
        """
        N, M = c2w.shape[0], resolution**2
        cam_locs_world = c2w[:, :3, 3]
        fx = intr[:, 0, 0]
        fy = intr[:, 1, 1]
        cx = intr[:, 0, 2]
        cy = intr[:, 1, 2]
        sk = intr[:, 0, 1]

        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device=c2w.device), \
                                        torch.arange(resolution, dtype=torch.float32, device=c2w.device), indexing='ij')) \
                                        * (1./resolution) + (0.5/resolution)
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0) # (resolution**2, 2)
        uv = uv.unsqueeze(0).repeat(N, 1, 1) # (batch, resolution**2, 2)

        x = uv[:, :, 0].view(N, -1)
        y = uv[:, :, 1].view(N, -1)
        z = torch.ones((N, M), device=c2w.device)

        x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1) - \
                  sk.unsqueeze(-1) * y / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
        y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

        cam_rel_points = torch.stack((x_lift, y_lift, z, torch.ones_like(z)), dim=-1)
        world_rel_points = torch.bmm(c2w, cam_rel_points.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

        rays_d = world_rel_points - cam_locs_world.unsqueeze(1).repeat(1, world_rel_points.shape[1], 1)
        rays_d = torch.nn.functional.normalize(rays_d, dim=2)

        rays_o = cam_locs_world.unsqueeze(1).repeat(1, rays_d.shape[1], 1)

        return rays_o, rays_d
    

class ImportranceRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = self.generate_planes()

    def generate_planes(self):
        """
        Defines planes by the three vectors that form the "axes" of the
        plane. Should work with arbitrary number of planes and planes of
        arbitrary orientation.
        """
        return torch.tensor([[[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]],
                             [[1, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0]],
                             [[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]]], dtype=torch.float32)

    def forward(self, features, decoder, rays_o, rays_d, op):
        self.plane_axes = self.plane_axes.to(rays_o.device)
        depths_coarse = self.sample_stratified(rays_o, op['ray_start'], op['ray_end'],
                                               op['N_samples'],
                                               op['is_disparity']) # (batch, N_rays, N_samples, 1)
        
        B, N_rays, N_samples, _ = depths_coarse.shape

        # coarse pass
        samples_xyz = (rays_o.unsqueeze(-2) + depths_coarse * rays_d.unsqueeze(-2)).reshape(B, -1, 3) # (batch, N_rays * N_samples, 3)
        samples_dir = rays_d.unsqueeze(-2).expand(-1, -1, N_samples, -1).reshape(B, -1, 3) # (batch, N_rays * N_samples, 3)

        out = self.run_model(features, decoder, samples_xyz, samples_dir, op)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(B, N_rays, N_samples, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(B, N_rays, N_samples, 1)

        # fine pass
        N_importances = op['N_importances']
        if N_importances > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, op)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importances)

            samples_xyz = (rays_o.unsqueeze(-2) + depths_fine * rays_d.unsqueeze(-2)).reshape(B, -1, 3) # (batch, N_rays * N_samples, 3)
            samples_dir = rays_d.unsqueeze(-2).expand(-1, -1, N_importances, -1).reshape(B, -1, 3) # (batch, N_rays * N_importances, 3)

            out = self.run_model(features, decoder, samples_xyz, samples_dir, op)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(B, N_rays, N_importances, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(B, N_rays, N_importances, 1)
            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                       depths_fine, colors_fine, densities_fine)
            
            # aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, op)

        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, op)

        return rgb_final, depth_final, weights.sum(2)

    def run_model(self, features, decoder, samples_xyz, samples_dir, op):
        sampled_features = self.sample_from_planes(self.plane_axes, features, samples_xyz, box_warp=op['box_warp'])
        return decoder(sampled_features, samples_dir)

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def project_onto_planes(self, planes, coordinates):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape N, M, 3
        # returns projections of shape N*n_planes, M, 2
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = planes.shape
        coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
        inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
        projections = torch.bmm(coordinates, inv_planes)
        return projections[..., :2]

    def sample_from_planes(self, plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape
        plane_features = plane_features.view(N*n_planes, C, H, W)

        coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

        projected_coordinates = self.project_onto_planes(plane_axes, coordinates).unsqueeze(1)
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
        return output_features

    def sample_from_3dgrid(self, grid, coordinates):
        """
        Expects coordinates in shape (batch_size, num_points_per_batch, 3)
        Expects grid in shape (1, channels, H, W, D)
        (Also works if grid has batch size)
        Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
        """
        batch_size, n_coords, n_dims = coordinates.shape
        sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                           coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                           mode='bilinear', padding_mode='zeros', align_corners=False)
        N, C, H, W, D = sampled_features.shape
        sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
        return sampled_features

    def sample_stratified(self, rays_o, ray_start, ray_end, N_samples, is_disparity=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = rays_o.shape
        if is_disparity:
            depths_coarse = torch.linspace(0, 1, 
                                           N_samples,
                                           device=rays_o.device).reshape(1, 1, N_samples, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(N_samples - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            depths_coarse = torch.linspace(ray_start, ray_end,
                                           N_samples,
                                           device=rays_o.device).reshape(1, 1, N_samples, 1).repeat(N, M, 1, 1)
            depth_delta = (ray_end - ray_start)/(N_samples - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse # (batch, N_rays, N_samples, 1)
    
    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals
    
    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples


class MipRayMarcher2(nn.Module):
    """
    The ray marcher takes the raw output of the implicit representation and
    uses the volume rendering equation to produce composited colors and depths.
    Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
    """
    def __init__(self):
        super().__init__()


    def run_forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2


        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights


    def forward(self, colors, densities, depths, rendering_options):
        composite_rgb, composite_depth, weights = self.run_forward(colors, densities, depths, rendering_options)

        return composite_rgb, composite_depth, weights