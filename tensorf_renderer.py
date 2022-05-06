######################################################
# THIS CODE IS DEPRECATED. 
# USERS SHOULD IMPLEMENT THEIR OWN RENDERING SCRIPT.
# MORE DETAILS TBD.
######################################################

# import torch

# import tensorfvis
# from TensoRF.dataLoader import dataset_dict
# from TensoRF.models import tensoRF
# import TensoRF.utils as trf_utils


# DEVICE_BACKEND = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DECOMP_METHODS = {
#     "CP": tensoRF.TensorCP,
#     "VM": tensoRF.TensorVM,
#     "VMSplit": tensoRF.TensorVMSplit,
# }


# def _init_tensorf(checkpoint_path="tensorf_engine_VM_weights.th", decomp_method="VMSplit"):
#         # A: grab the checkpoint file
#         ckpt = torch.load(checkpoint_path, map_location=DEVICE_BACKEND)
#         kwargs = ckpt["kwargs"]
#         kwargs.update({"device": DEVICE_BACKEND})
#         # B: instantiate the appropiate TensoRF
#         tensorf = DECOMP_METHODS[decomp_method](**kwargs)
#         tensorf.load(ckpt)
#         return tensorf


# if __name__ == "__main__":
#     # A: load in our TensoRF, and the dataset
#     tensorf = _init_tensorf()
#     dataset = dataset_dict["blender"]  # comes from the "dataset_name" in our config
#     test_dataset = dataset(
#         "./data/nerf_synthetic/engine_6_ds",
#         split="test",
#         downsample=1.0,  # I guess I used the default value for "downsample_test"
#         is_stack=True
#     )
#     # B: make a new Scene
#     scene = tensorfvis.Scene("TensoRF Real-time Renderer, V0.0.1")
#     scene.add_axes()
#     # C: set TensoRF as the rendering algorithm
#     scene.set_nerf(
#         nerf_func, 
#         center=[0.0, 0.0, 0.0], 
#         radius=1.5, 
#         use_dirs=True,
#         dirs={}
#     )
#     # D: render!
#     scene.display(port=8899)