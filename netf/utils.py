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
  flags.DEFINE_integer("save_every", 10000,
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


def render_volume(model, variables, volume_box, resolutions, prefix):
  """Render the volme
  Args: 
    model: NeTF model.
    variables: trained variables.
    volume_box: the bounding box for the volume.
    resolutions: the resolution for rendering (nx, ny, nz, length)
  """
  from .models_utils import posenc
  cx, cy, cz, length = volume_box
  xv = np.linspace(cx-length, cx+length, resolutions[0])
  yv = np.linspace(cx-length, cx+length, resolutions[1])
  zv = np.linspace(cx-length, cx+length, resolutions[2])
  xv, yv, zv = np.meshgrid(xv, yv, zv)
  points = np.stack([xv, yv, zv], axis=-1)
  views = np.zeros((*resolutions, 2))
  views[..., 0] = np.pi / 2
  views[..., 1] = np.pi / 2
  points = points.reshape(-1, 3)
  views = views.reshape(-1, 2)
  raw_sigma = np.empty((len(points), 1, 1), dtype=np.float)
  raw_rho = np.empty((len(points), 1, 1), dtype=np.float)
  normal = np.empty((len(points), 1, 3), dtype=np.float)
  stride = 64**3

  def get_norm(points, views):
      def forward(points, views):
        enc_points = posenc(points, model.min_deg_point, model.max_deg_point, True)
        enc_views = posenc(views, 0, model.deg_view, legacy_posenc_order=True)
        enc_views = enc_views.reshape(-1, 1, enc_views.shape[-1])
        enc_points = enc_points.reshape(-1, 1, enc_points.shape[-1])
        sigma = model.apply(variables, enc_points, enc_views, method=model.coarse_encode)[0]
        return sigma.flatten()[0]
    
      normal = jax.jit(jax.grad(forward))
      normal = jax.jit(jax.vmap(normal))
      return normal(points, views)

  for i in range(0, len(points), stride):
    enc_points = posenc(points[i:i+stride], model.min_deg_point, model.max_deg_point, True)
    enc_views = posenc(views[i:i+stride], 0, model.deg_view, legacy_posenc_order=True)
    enc_views = enc_views.reshape(-1, 1, enc_views.shape[-1])
    enc_points = enc_points.reshape(-1, 1, enc_points.shape[-1])
    raw_sigma[i:i+stride], raw_rho[i:i+stride] = model.apply(variables, enc_points, enc_views, method=model.coarse_encode)
    print("render {}/{} ...".format(i//stride, len(points)//stride))

  import matplotlib.pyplot as plt
  zv, yv = np.meshgrid(np.arange(256), np.arange(256))
  raw_sigma = raw_sigma.reshape(resolutions)
  
  
  # root finding
  pdb.set_trace()
  index = raw_sigma.argmax(axis=0)
  xoy = (raw_sigma).max(axis=0)
  assert (raw_sigma[index, yv, zv] == xoy).all()
  query_point = points.reshape(*resolutions, 3)[index, yv, zv, :]
  query_view = np.zeros((256, 256, 2))
  query_view[..., 0] = np.pi / 2
  query_view[..., 1] = np.pi / 2
  normals = get_norm(query_point.reshape(-1, 3), query_view.reshape(-1, 2))
  normals = normals / (np.sqrt((normals**2).sum(-1, keepdims=True)) + 1e-8)
  normals = normals.reshape(256, 256, 3)

  import matplotlib.pyplot as plt
  import matplotlib
  import matplotlib.cm as cmx
  index = np.where(raw_sigma >= 0.5)
  fig = plt.figure(figsize=(32, 32))
  ax = plt.axes(projection='3d')
  cm = plt.get_cmap('Greys')
  cNorm = matplotlib.colors.Normalize(vmin=raw_sigma[index].min(), vmax=raw_sigma[index].max())
  scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
  # ax.scatter(index[index!=0], yv[index!=0], zv[index!=0], c=scalarMap.to_rgba(xoy[index!=0].flatten()))
  colors = scalarMap.to_rgba(raw_sigma[index].flatten())
  colors[...,-1] = raw_sigma[index].flatten() / raw_sigma[index].max()

  # ax.scatter(index[0], index[1], index[2], c=colors)
  # ax.scatter(index[0], index[1], index[2], c=scalarMap.to_rgba(index[0].flatten()))
  # for ii in range(0, 361, 30):
  #   ax.view_init(elev=10, azim=ii)
  #   plt.savefig('views/{}_angle_{}.jpg'.format(prefix, ii))
  from mayavi import mlab
  mlab.points3d(index[0], index[1], index[2], color=(0,1,0), scale_factor=0.01)
  mlab.savefig('test_mlab.jpg')
  pdb.set_trace()

  plt.imshow(normals[..., 0])
  plt.colorbar()
  plt.savefig('{}_norm.png'.format(prefix))
  plt.close()
  pdb.set_trace()
  plt.imshow(np.array(xoy))
  plt.colorbar()
  plt.savefig('{}_sigma.png'.format(prefix))
  plt.close()
  raw_rho = raw_rho.reshape(resolutions)
  xoy = (raw_rho).max(axis=0)
  plt.imshow(np.array(xoy))
  plt.colorbar()
  plt.savefig('{}_rho.png'.format(prefix))
  plt.close()
  xoy = (raw_rho*raw_sigma).max(axis=0)
  plt.imshow(np.array(xoy))
  plt.colorbar()
  plt.savefig('{}_albedo.png'.format(prefix))
  plt.close()