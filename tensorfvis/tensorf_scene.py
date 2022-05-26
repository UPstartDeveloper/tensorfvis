from typing import (
    Any, 
    Callable,
    List,
    Optional,
    Tuple,
    Union
)

import numpy as np

from .scene import Scene

class TensorfScene(Scene):
    def set_nerf(
        self,
        rgb_sigma_path: str,
        eval_fn: Callable[..., Tuple[Any, Any]],
        center: Union[
            Tuple[float, float, float], List[float], float, np.ndarray, None
        ] = None,
        radius: Union[
            Tuple[float, float, float], List[float], float, np.ndarray, None
        ] = None,
        scale: float = 1.0,
        reso: int = 256,
        use_dirs: bool = False,
        use_tensorf: bool = False,
        sh_deg: int = 1,
        sh_proj_sample_count: int = 15,
        sh_proj_use_sparse: bool = True,
        sigma_thresh: float = 3.0,
        weight_thresh: float = 0.001,
        r: Optional[Any] = None,
        t: Optional[Any] = None,
        focal_length: Optional[Union[float, Tuple[float, float]]] = None,
        image_width: Optional[float] = None,
        image_height: Optional[float] = None,
        sigma_multiplier: float = 1.0,
        chunk: int = 720720,
        device: str = "cuda:0",
    ):
        """
        TODO[refactor-later]: cleanup based on what params are really needed

        Discretize and display a NeRF (low quality, for visualization purposes only).
        Currently only supports PyTorch NeRFs.
        Requires tqdm, torch, svox, scipy

        :param rgb_sigma_path: str of Path-like object, that points to the RGB+density values
                        to use weight thresholding.

        :param eval_fn:
                        - If :code:`use_dirs=False`: NeRF function taking a batch of points :code:`(B, 3)` and returning :code:`(rgb (B, 3), sigma (B, 1))` after activation applied.

                        - If :code:`use_dirs=True` then this function should take points :code:`(B, 1, 3)` and kwarg 'dirs' :code:`(1, sh_proj_sample_count, 3)`; it should return :code:`(rgb (B, sh_proj_sample_count, 3), sigma (B, 1))` sigma activation
                            should be applied but rgb must NOT have activation applied for SH projection to work correctly.

        :param center: float or (3,), xyz center of volume to discretize
                       (will try to infer from cameras from add_camera_frustum if not given)
        :param radius: float or (3,), xyz half edge length of volume to discretize
                       (will try to infer from cameras from add_camera_frustum if not given)
        :param scale: float, multiples radius by this before using it (this is provided
                      for convenience, for manually scaling the scene boundaries which
                      is often needed to get the right bounds if using automatic
                      radius from camera frustums i.e. not specifying center and radius)
        :param reso: int, resolution of tree in all dimensions (must be power of 2)
        :param use_dirs: bool, if true, assumes normal NeRF with viewdirs; uses SH projection
                         to recover SH at each point with degree sh_deg.
        :param use_tensorf: bool, if true, will assume use of TensoRF's own utilities to 
                         evaluate your NeRF on the grid. use_dirs should be false.
                         Please see this TensoRF repo for more details: 
                         https://github.com/UPstartDeveloper/NeRF-to-XR/blob/skateboard-api/snerg/model_zoo/tensorf.py.
        :param sh_deg: int, SH degree if use_dirs, must be between 0-4
        :param sh_proj_sample_count: SH projection samples if use_dirs
        :param sh_proj_use_sparse: Use sparse SH projection via least-squares rather than
                                   monte carlo inner product
        :param sigma_thresh: float, simple density threshold (used if r, t not given)
        :param weight_thresh: float, weight threshold as in PlenOctrees (used if r, t given)
        :param r: (N, 3) or (N, 4) or (N, 3, 3) or None, optional
                  C2W rotations for each camera, either as axis-angle,
                  xyzw quaternion, or rotation matrix; if not given, only one camera
                  is added at identity.
        :param t: (N, 3) or None, optional
                  C2W translations for each camera applied after rotation;
                  if not given, only one camera is added at identity.
        :param focal_length: float or Tuple (fx, fy), optional, focal length for weight thresholding
        :param image_width: float, optional, image width for weight thresholding
        :param image_height: float, optional, image height for weight thresholding
        """
        # TODO[later]: replace the magic numbers with config vars or constants
        # TODO[later]: use if statements wherever .cuda() is used here

        import torch

        # Sets params for the NeRF
        if center is None and not np.isinf(self.bb_min).any():
            center = (self.bb_min + self.bb_max) * 0.5
        if radius is None and not np.isinf(self.bb_min).any():
            radius = (self.bb_max - self.bb_min) * 0.5

        if isinstance(center, list) or isinstance(center, tuple):
            center = np.array(center)
        elif isinstance(center, torch.Tensor):
            center = center.clone().detach()
        if isinstance(radius, list) or isinstance(radius, tuple):
            radius = np.array(radius)
        radius *= scale
        self._update_bb(center - radius)
        self._update_bb(center + radius)

        print("* Discretizing NeRF (requires torch, tqdm, svox, scipy)")

        # Determine how to move from camera 2 world coords, if applicable
        if r is not None and t is not None:
            c2w = np.eye(4, dtype=np.float32)[None].repeat(r.shape[0], axis=0)
            c2w[:, :3, 3] = t
            c2w[:, :3, :3] = _scipy_rotation_from_auto(r).as_matrix()
            c2w = torch.from_numpy(c2w).to(device=device)
        else:  # no rotations or translations going on
            c2w = None

        from tqdm import tqdm
        from svox import N3Tree
        from svox.helpers import _get_c_extension
        from .sh import project_function_sparse, project_function

        project_fun = (
            project_function_sparse if sh_proj_use_sparse else project_function
        )

        # Let's get the PyTorch CUDA extension for PlenOctrees
        _C = _get_c_extension()
        with torch.no_grad():
            sh_dim = (sh_deg + 1) ** 2
            data_format = f"SH{sh_dim}" if use_dirs else "RGBA"
            init_grid_depth = reso.bit_length() - 2

            # Guard Clause
            assert 2 ** (init_grid_depth + 1) == reso, "Grid size must be a power of 2"

            # Init the PlenOctree
            tree = N3Tree(
                N=2,
                init_refine=0,
                init_reserve=500000,
                geom_resize_fact=1.0,
                depth_limit=init_grid_depth,
                radius=radius,
                center=center,
                data_format=data_format,
                device=device,
            )
            offset = tree.offset.cpu()
            scale = tree.invradius.cpu()

            # Init the grid
            arr = (torch.arange(0, reso, dtype=torch.float32) + 0.5) / reso
            xx = (arr - offset[0]) / scale[0]
            yy = (arr - offset[1]) / scale[1]
            zz = (arr - offset[2]) / scale[2]
            # dims here are [reso ** 3, 3]
            grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T

            print("Evaluating NeRF on a grid - loading serialized RGB + sigma")
            rgb_sigma = torch.load(rgb_sigma_path)

            def _calculate_grid_weights(
                sigmas, c2w, focal_length, image_width, image_height
            ):
                print("  Performing weight thresholding ")
                # Weight thresholding impl
                opts = _C.RenderOptions()
                opts.step_size = 1e-5
                opts.sigma_thresh = 0.0
                opts.ndc_width = -1

                cam = _C.CameraSpec()
                if isinstance(focal_length, float):
                    focal_length = (focal_length, focal_length)
                cam.fx = focal_length[0]
                cam.fy = focal_length[1]
                cam.width = image_width
                cam.height = image_height

                grid_data = sigmas.reshape((reso, reso, reso)).contiguous()
                maximum_weight = torch.zeros_like(grid_data)
                camspace_trans = torch.diag(
                    torch.tensor(
                        [1, -1, -1, 1], dtype=sigmas.dtype, device=sigmas.device
                    )
                )
                for idx in tqdm(range(c2w.shape[0])):
                    cam.c2w = c2w[idx]
                    cam.c2w = cam.c2w @ camspace_trans
                    grid_weight, _ = _C.grid_weight_render(
                        grid_data, cam, opts, tree.offset, tree.invradius,
                    )
                    maximum_weight = torch.max(maximum_weight, grid_weight)
                return maximum_weight

            if c2w is None:
                # Sigma thresh
                mask = rgb_sigma[..., -1] >= sigma_thresh
            else:
                # Weight thresh
                assert (
                    focal_length is not None
                    and image_height is not None
                    and image_width is not None
                ), (
                    "All of r, t, focal_length, image_width, "
                    "image_height should be provided to set_nerf to use weight thresholding"
                )
                grid_weights = _calculate_grid_weights(
                    rgb_sigma[..., -1:],
                    c2w.float(),
                    focal_length,
                    image_width,
                    image_height,
                )
                mask = grid_weights.reshape(-1) >= weight_thresh
            grid = grid[mask]
            rgb_sigma = rgb_sigma[mask]
            del mask
            assert (
                grid.shape[0] > 0
            ), "This NeRF is completely empty! Make sure you set the bounds reasonably"
            print(
                "  Grid shape =",
                grid.shape,
                "min =",
                grid.min(dim=0).values,
                " max =",
                grid.max(dim=0).values,
            )
            grid = grid.cuda()

            torch.cuda.empty_cache()

            print("Building octree structure")
            for i in range(init_grid_depth):
                tree[grid].refine()
            print("  tree:", tree)

            if sigma_multiplier != 1.0:
                rgb_sigma[..., -1] *= sigma_multiplier
            tree[grid] = rgb_sigma

            # Just a sanity check, if it failed maybe all points got filtered out
            assert tree.max_depth == init_grid_depth
            print("Finishing up")

            tree.shrink_to_fit()
            self.nerf = tree