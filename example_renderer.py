######################################################
# THIS SCRIPT IS IN PROGRESS!
# It is based on using TensoRF, 
# described by Chen et. al.: https://arxiv.org/abs/2203.09517
######################################################


from absl import app
from absl import flags

import torch
# import tensorflow as tf
from jax import random
import numpy as np

from snerg.tensorfvis import Scene

from snerg import model_zoo
from snerg.model_zoo import utils, datasets


FLAGS = flags.FLAGS
utils.define_flags()


def main(unused_argv):

    ### HELPERS
    def _get_model_and_ds():
        # tf.config.experimental.set_visible_devices([], "GPU")
        # tf.config.experimental.set_visible_devices([], "TPU")

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

    def _infer_on_rays(xyz_sampled, origins, dirs):
        """relies on the outer scope to use TensoRF (requires TensoRF trained with SH rendering)."""
        # A: get the sigma
        rays_chunk = torch.cat([origins, dirs], dim=1)  # [sh_proj_sample_count, 6]
        sigma = tensorf.compute_density(rays_chunk)  # 
        # ensure it's 2D
        if len(sigma.shape) == 1:
            sigma = sigma[np.newaxis, :]
        # B: get the rgb
        features = tensorf.compute_feature(xyz_sampled)
        raw_rgb = tensorf.compute_raw_rgb(dirs, features)

        return raw_rgb, sigma

    ### DRIVER
    # A: load in our TensoRF, and the dataset
    tensorf, _, dataset = _get_model_and_ds()
    rotations, translations = None, None
    if isinstance(dataset, datasets.Blender):
        camtoworlds = dataset.peek()["camtoworlds"]  # dims are (4, 4)
        rotations, translations = datasets.decompose_camera_transforms(camtoworlds)
        # ensure rotations is a 3D array and translations is 2D
        rotations = rotations[np.newaxis, :, :]  # (1, 3, 3)
        translations = translations[np.newaxis, :]  # (1, 3)
    # B: make a new Scene
    scene = Scene("TensoRF Real-time Renderer, Version 0.1")
    scene.add_axes()
    # C: set TensoRF as the rendering algorithm
    scene.set_nerf(
        _infer_on_rays,
        center=tensorf.get_center().cpu(),
        # radius=2,  # ball park guess for now, using what radius a sphere might have if you put in the box
        use_dirs=True,
        device=tensorf.DEVICE_BACKEND,
        sh_deg=2,  # TODO: remove this magic number with a property of tensorf (have to first refactor SHRender)
        reso=512,  # power of two that's closest to "voxel_resolution" flag on config, but won't cause OOM
        # scale=25,  # using the same value as the "distance_scale" attr of TensoRF
        sh_proj_sample_count=9,  # TODO[do-better]: hardcoded after playing around - read https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
        r=rotations, t=translations,
        focal_length=dataset.focal,
        image_height=dataset.h, image_width=dataset.w,
        # chunk=81,
    )
    # D: render!
    scene.display(port=8899)


if __name__ == "__main__":
    app.run(main)
