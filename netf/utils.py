import collections
import os
from os import path
from sys import prefix
from jax._src.api import grad, jacfwd
from matplotlib.pyplot import axis
import yaml
from absl import flags
import flax
import numpy as np
import jax.numpy as jnp
import pdb
import jax
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
# __all__ = ['define_flags', "update_flags"]

INTERNAL = False
BASE_DIR = "configs"

@flax.struct.dataclass
class TrainState:
  optimizer: flax.optim.Optimizer

@flax.struct.dataclass
class Stat:
  loss: float

# Transiences = collections.namedtuple("Transiences", ("origins", "radius"))

def namedtuple_map(fn, tup):
  """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
  return type(tup)(*map(fn, tup))


def define_flags():
  """Define flags for both training and evaluation modes."""
  # Dataset Flags
  flags.DEFINE_string("data_path", None, "input data directory")
  flags.DEFINE_string("cache_dir", 'cache', "where to store the ckpts and logs")
  flags.DEFINE_string("config", None, "using the config file")
  flags.DEFINE_enum("batching", "all_grids", ["all_grids"], "how to batching the input")
  flags.DEFINE_integer("batch_size", 128, "the number of hist in a mini-batch")
  
  # Model Flags
  flags.DEFINE_integer("min_deg_point", 0,
                      "Minimum degree of positional encoding for points.")
  flags.DEFINE_integer("max_deg_point", 10,
                      "Maximum degree of positional encoding for points.")
  flags.DEFINE_integer("deg_view", 4,
                      "Degree of positional encoding for viewdirs.")
  flags.DEFINE_string("net_activation", "relu",
                      "activation function used in the MLP.")
  flags.DEFINE_string("rho_activation", "relu",
                      "activation function used to produce reflactance.")
  flags.DEFINE_string("sigma_activation", "relu",
                      "activation function used to produce density.")
  flags.DEFINE_integer("num_coarse_samples", 64*64, 
                        "the number of samples on each hemisphere for coarse model.")
  flags.DEFINE_integer("num_fine_samples", 32*32, 
                        "the number of samples on each hemisphere for fine model.")
  flags.DEFINE_float("reg_normal_eps", 1e-3, "eps to regularize normal consistence.")
  flags.DEFINE_bool("cond_normal", False, "condition on surface normals.")
  flags.DEFINE_bool("occlusion", False, "take occlusion into consideration.")
  flags.DEFINE_float("reg_coeff", 0.0, "normal consistence loss.")
  # Training
  flags.DEFINE_float("lr_init", 1e-4, "The initial learning rate")
  flags.DEFINE_float("lr_final", 1e-6, "The final learning rate")
  flags.DEFINE_float("lr_delay_steps", 0, "the number of steps at the begining of "
        "training to reduce the learning rate by lr_delay_mult")
  flags.DEFINE_float("lr_delay_mult", 1, "A multiplier on the learning rate when the step"
        "is < lr_delay_steps")
  flags.DEFINE_float("grad_max_norm", 0.,
                    "The gradient clipping magnitude (disabled if == 0).")
  flags.DEFINE_float("grad_max_val", 0.,
                    "The gradient clipping value (disabled if == 0).")
  flags.DEFINE_integer("max_steps", 1000000,
                      "the number of optimization steps.")
  flags.DEFINE_integer("save_every", 50000,
                      "the number of steps to save a checkpoint.")
  flags.DEFINE_integer("test_every", 20000,
                      "the number of steps to save a checkpoint.")
  flags.DEFINE_integer("print_every", 100,
                      "the number of steps between reports to tensorboard.")
  flags.DEFINE_integer("gc_every", 1000,
                      "the number of steps to run python garbage collection.")




def update_flags(args):
    """Update the flags in `args` with the contents of the config YAML file."""
    pth = path.join(BASE_DIR, args.config + ".yaml")
    with open_file(pth, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Only allow args to be updated if they already exist.
    invalid_args = list(set(configs.keys()) - set(dir(args)))
    if invalid_args:
        raise ValueError(f"Invalid args {invalid_args} in {pth}.")
    args.__dict__.update(configs)


def open_file(pth, mode="r"):
  if not INTERNAL:
    return open(pth, mode=mode)


def file_exists(pth):
  if not INTERNAL:
    return path.exists(pth)


def listdir(pth):
  if not INTERNAL:
    return os.listdir(pth)


def isdir(pth):
  if not INTERNAL:
    return path.isdir(pth)


def makedirs(pth):
  if not INTERNAL:
    os.makedirs(pth)


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.
  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.
  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.
  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = np.clip(step / max_steps, 0, 1)
  log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
  return delay_rate * log_lerp

def root_finding(ray, thresh):
  """
  finding the first intersection point between surface and ray.
  Args:
    ray: a straight ray, i.e. a list of points.
    tresh: threshold for the surface. 
  """
  pdb.set_trace()
  return ((ray[:, :-1] < 0.5) * (ray[:, 1:] >= 0.5)).argmax(-1)
  # ind = jnp.where(ray[:, :-1] < 0.5 and ray[:, 1:] >= 0.5)
  # if len(ind) == 0:
    # return -1
  # return ind.min()



def rendering(model, variables, volume_box, resolutions, prefix):
  """Render the volme
  Args: 
    model: NeTF model.
    variables: trained variables.
    volume_box: the bounding box for the volume.
    resolutions: the resolution for rendering (nx, ny, nz, length)
  """
  from .models_utils import posenc
  cx, cy, cz, length = volume_box
  xs = np.linspace(cx-length, cx+length, resolutions[0])
  ys = np.linspace(cy-length, cy+length, resolutions[1])
  zs = np.linspace(cz-length, cz+length, resolutions[2])
  xv, yv, zv = np.meshgrid(xs, ys, zs)
  points = np.stack([xv, yv, zv], axis=-1)
  views = np.zeros((*resolutions, 2))
  views[..., 0] = np.pi / 2
  views[..., 1] = -np.pi / 2
  points = points.reshape(-1, 3)
  views = views.reshape(-1, 2)
  raw_sigma = np.empty((len(points), 1, 1), dtype=np.float)
  raw_rho = np.empty((len(points), 1, 1), dtype=np.float)
  stride = 64**3

  for i in tqdm(range(0, len(points), stride)):
    vs = views[i:i+stride]
    pts = points[i:i+stride]
    if model.cond_normal:
      normals = model.apply(variables, pts, method=model.get_norm)
      vs = jnp.concatenate((vs, normals), axis=-1)
    enc_points = posenc(pts, model.min_deg_point, model.max_deg_point, True)
    enc_views = posenc(vs, 0, model.deg_view, legacy_posenc_order=True)
    enc_views = enc_views.reshape(-1, 1, enc_views.shape[-1])
    enc_points = enc_points.reshape(-1, 1, enc_points.shape[-1])
    raw_sigma[i:i+stride], raw_rho[i:i+stride] = model.apply(variables, enc_points, enc_views, method=model.coarse_encode)

  raw_sigma = raw_sigma.reshape(resolutions)
  raw_rho = raw_rho.reshape(resolutions)
  index = np.where(raw_sigma >= 0.9)
  if len(index[0]) > 0:
    query_points = points.reshape(*resolutions, 3)[index]
    sample_index = np.random.choice(len(query_points), 1500)
    sample_points = query_points[sample_index]
    sample_normals = model.apply(variables, sample_points, method=model.get_norm)
    ax = plt.axes(projection='3d')
    ax.quiver(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], sample_normals[:, 0], sample_normals[:,1], sample_normals[:,2], length=0.03, linewidth=0.5)
    ax.view_init(elev=10, azim=30)
    plt.savefig('{}_normal_30'.format(prefix), dpi=1000)
    ax.view_init(elev=10, azim=90)
    plt.savefig('{}_normal_90'.format(prefix), dpi=1000)
    ax.view_init(elev=10, azim=210)
    plt.savefig('{}_normal_210'.format(prefix), dpi=1000)
    ax.view_init(elev=10, azim=270)
    plt.savefig('{}_normal_270'.format(prefix), dpi=1000)
    plt.close()

    cm = plt.get_cmap('Greys')
    cNorm = matplotlib.colors.Normalize(vmin=raw_sigma[index].min(), vmax=raw_sigma[index].max())
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    ax = plt.axes(projection='3d')
    
    if len(query_points) > 100000:
      query_points = sample_points
    colors = scalarMap.to_rgba(raw_sigma[index].flatten())
    colors[...,-1] = (raw_sigma[index].flatten()-raw_sigma[index].min()) / raw_sigma[index].max()
    # ax.scatter(query_points[:, 0], query_points[:, 1], query_points[:, 2], c=colors)
    # ax.scatter(query_points[:, 0], query_points[:, 1], query_points[:, 2])
    ax.scatter(xs[index[0]], ys[index[1]], zs[index[2]], c=colors)
    # ax.scatter(index[0], index[1], index[2], c=colors)
    ax.view_init(elev=10, azim=30)
    plt.savefig('{}_surface_30'.format(prefix), dpi=1000)
    ax.view_init(elev=10, azim=90)
    plt.savefig('{}_surface_90'.format(prefix), dpi=1000)
    ax.view_init(elev=10, azim=210)
    plt.savefig('{}_surface_210'.format(prefix), dpi=1000)
    ax.view_init(elev=10, azim=270)
    plt.savefig('{}_surface_270'.format(prefix), dpi=1000)
    plt.clf()
    plt.close()


  plt.imshow(raw_sigma.max(axis=0))
  plt.colorbar()
  plt.savefig('{}_YOZ_sigma'.format(prefix), dpi=1000)
  plt.clf()
  plt.imshow(raw_rho.max(axis=0))
  plt.colorbar()
  plt.savefig('{}_YOZ_rho'.format(prefix), dpi=1000)
  plt.clf()
  plt.imshow(raw_rho.max(axis=0) * raw_sigma.max(axis=0))
  plt.colorbar()
  plt.savefig('{}_YOZ_abedlo'.format(prefix), dpi=1000)
  plt.close()