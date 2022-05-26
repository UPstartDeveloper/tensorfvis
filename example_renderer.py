######################################################
# THIS SCRIPT IS IN PROGRESS!
# It is based on using TensoRF,
# described by Chen et. al.: https://arxiv.org/abs/2203.09517
######################################################

# always include these imports
import torch

import tensorflow as tf
from jax import random
import numpy as np
from tensorfvis.scene import Scene

# optional to include gc - might be useful to avoid OOM
import gc

# model-specific imports - differs based on what kind of radiance field tooling you have
# (in this case, we have SNeRG + TensoRF)
from absl import app, flags

from snerg import model_zoo
from snerg.model_zoo import utils, datasets

from TensoRF.dataLoader import ray_utils
import TensoRF.dataLoader as trf_data_utils
from TensoRF.models import tensoRF
from TensoRF import renderer


DEVICE_BACKEND = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS = flags.FLAGS
utils.define_flags()


def main(unused_argv):

    ### HELPERS
    def _get_trf_dataset(FLAGS):
        """Returns the in-memory PyTorch version of the dataset for this TensoRF."""
        # assues self.dataset is one of ["blender", "llff", "tankstemple", "nsvf", "own_data"]
        data_loader = trf_data_utils.dataset_dict[FLAGS.dataset]
        # get the test set
        print(f"FLAGS: {FLAGS.absolute_data_dir, FLAGS.tensorf_factor}")
        return data_loader(
            FLAGS.absolute_data_dir, split="test", downsample=FLAGS.tensorf_factor, is_stack=True,
        )

    def _get_model_and_ds():
        tf.config.experimental.set_visible_devices([], "GPU")
        tf.config.experimental.set_visible_devices([], "TPU")

        rng = random.PRNGKey(20200823)

        if FLAGS.config is not None:
            utils.update_flags(FLAGS)
        FLAGS.train_dir = "/home/ec2-user/NeRF-to-XR/TensoRF/log/tensorf_lego_VM"
        FLAGS.data_dir = "/home/ec2-user/NeRF-to-XR/engine_6_ds"

        # get the JAX wrapper class around TensoRF
        test_dataset = datasets.get_dataset("test", FLAGS)
        rng, key = random.split(rng)
        model, init_variables = model_zoo.get_model(key, test_dataset.peek(), FLAGS)

        return model, init_variables, test_dataset

    def _torch_get_model_and_ds(FLAGS):
        '''only using PyTorch
        TODO[refactor] - make a function to do this in TensoRF.models.__init__
        '''
        if FLAGS.config is not None:
            utils.update_flags(FLAGS)

        # A: grab the checkpoint file
        ckpt = torch.load(FLAGS.tensorf_checkpoint, map_location=DEVICE_BACKEND)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": DEVICE_BACKEND})
        # B: instantiate the appropiate TensoRF
        tensorf = tensoRF.TensorVMSplit(**kwargs)
        tensorf.load(ckpt)
        return tensorf

    def _infer_on_rays(xyz_sampled=None, dirs=None, sh_deg=2):
        """
        relies on the outer scope to use TensoRF (requires TensoRF trained with SH rendering).
        
        Dims:
            xyz_sampled: [chunk_size  // sh_proj_sample_count, 1, 3]
            dirs: [sh_proj_sample_count, 3]
            sh_deg: int, for TensoRF you can use 0-4
        """
        # TODO: for c2w - use an identity matrix for now
        # c2w = torch.ones((4, 4), device=DEVICE_BACKEND)[:3, :]  # (3, 4) matrix
        # directions = test_ds.directions.to(device=DEVICE_BACKEND)
        #  for directions - pass the directions obj from the ds_obj
        # rays_origin, rays_dir = ray_utils.get_rays(directions, c2w)
        # concat THOSE rays together, to pass to the subsequent funcs
        # rays_chunk = torch.cat([rays_origin, rays_dir], dim=1)
        # A: get the sigma
        # sigma = tensorf.compute_density(rays_chunk, xyz_sampled.shape[0])
        # # B: get the rgb
        # features = tensorf.compute_feature(xyz_sampled)
        # raw_rgb = tensorf.compute_raw_rgb(dirs, features)
        # reshaped_rgb = raw_rgb.transpose(1, 2)  # TODO[check this has dims of: [batch_size, sh_proj_sample_count, 3]
        # return reshaped_rgb, sigma
        # option 1 - JAX style
        # return tensorf.evaluate_on_grid(test_ds)
        # TODO: PyTorch style - pass the chunk size
        rgb, sigma = renderer.evaluation_for_rgb_sigma(
            test_dataset=test_ds,
            tensorf=tensorf,
            c2ws=test_ds.poses,
            renderer=renderer.OctreeRender_trilinear_fast,
            white_bg=test_ds.white_bg,
            return_rgb_sigma_only=True,
            device=DEVICE_BACKEND
        )
        rgb_sigma = torch.cat([rgb, sigma], dim=-1)
        del rgb, sigma  # trying to avoid OOM
        return rgb_sigma

    ### DRIVER
    tf.config.experimental.set_visible_devices([], "GPU")
    tf.config.experimental.set_visible_devices([], "TPU")
    torch.cuda.set_per_process_memory_fraction(fraction=1.00)
    # A: load in our TensoRF, and the dataset
    # tensorf, _, jax_dataset = _get_model_and_ds()
    tensorf = _torch_get_model_and_ds(FLAGS)
    test_ds = _get_trf_dataset(FLAGS)
    # TODO[refactor] - consider using the TensoRF ds for getting r and t
    rotations, translations = None, None
    # if isinstance(jax_dataset, datasets.Blender):
    # camtoworlds = jax_dataset.peek()["camtoworlds"]  # dims are (4, 4)
    camtoworlds = test_ds.poses  # if going only w/ PyTorch
    rotations, translations = datasets.decompose_camera_transforms(camtoworlds)
    # ensure rotations is a 3D array and translations is 2D
    # rotations = rotations[np.newaxis, :, :]  # (1, 3, 3)
    translations = translations[np.newaxis, :]  # (1, 3)
    # del jax_dataset  # avoiding OOM
    gc.collect()
    # B: make a new Scene
    scene = Scene("TensoRF Real-time Renderer, Version 0.2")
    # scene.add_axes()
    # C: set TensoRF as the rendering algorithm

    # set the configs - these equations are just for hacking purposes
    sh_deg = 2
    sh_proj_sample_count = (sh_deg + 1) ** 2  # purely for implementation reasons
    reso = torch.tensor([2 ** 5])  # 10 - 1_024
    num_batches = torch.tensor([2 ** 10])
    R = int(torch.log2(reso))
    B = int(torch.log2(num_batches))
    chunk = sh_proj_sample_count * (2 ** ((3 * R) - B))
    scene.set_nerf(
        _infer_on_rays,
        # center=tensorf.get_center().cpu(),  # TODO[rerun] with test_ds.center
        center=test_ds.center,
        # TODO[renrun] use test_ds.radius
        radius=test_ds.radius,  # required when there's no previous call to _update_bb
        use_dirs=False,
        use_tensorf=True,
        device=DEVICE_BACKEND,
        sh_deg=sh_deg,  # guessing it's 2, based on how eval_sh() is used in TensoRF code
        reso=int(
            reso
        ),  # power of two that's closest to "voxel_resolution" flag on config, but won't cause OOM
        scale=tensorf.distance_scale,
        # scale=1.0,  # trying to avoid empty NeRF
        sh_proj_sample_count=sh_proj_sample_count,  # TODO[do-better]: hardcoded after playing around - read https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
        r=rotations,
        t=translations,
        focal_length=test_ds.focal,
        image_height=test_ds.img_wh[1],
        image_width=test_ds.img_wh[0],
        chunk=chunk,
    )
    # D: render!
    scene.display(port=8899)


if __name__ == "__main__":
    app.run(main)
