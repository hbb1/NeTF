# coding=utf-8
# Copyright 2021 ShanghaiTech University Authors.
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Helper functions/classes for model definition."""

import functools
from typing import Any, Callable

from flax import linen as nn
import jax
from jax import lax
from jax import random
from jax._src.api import jit
import jax.numpy as jnp
import numpy as np
import pdb

DEBUG=False

class MLP(nn.Module):
  """A simple MLP."""
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_condition: int = 1  # The depth of the second part of MLP.
  net_width_condition: int = 128  # The width of the second part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  skip_layer: int = 4  # The layer to add skip layers to.
  num_sigma_channels: int = 1  # The number of sigma channels.
  num_rho_channels: int = 1  # The number of rho channels.

  @nn.compact
  def __call__(self, x, condition=None):
    """ Evaluate the MLP
    Args:
    x: jnp.narray(float32), [batch_size, num_samples, features]
    condition: the view direction [batch_size, num_samples, features]
    Returns:
    hist: histgram
    """
    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])
    dense_layer = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform()
    )
    inputs = x
    for i in range(self.net_depth):
      x = dense_layer(self.net_width)(x)
      x = self.net_activation(x)
      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)
    
    raw_sigma = dense_layer(self.num_sigma_channels)(x).reshape(
                  [-1, num_samples, self.num_sigma_channels])

    if condition is not None:
      # Output of the first part of MLP.
      bottleneck = dense_layer(self.net_width)(x)
      condition = condition.reshape([-1, condition.shape[-1]])
      x = jnp.concatenate([bottleneck, condition], axis=-1)
      # Here use 1 extra layer to align with the original nerf model.
      for i in range(self.net_depth_condition):
        x = dense_layer(self.net_width_condition)(x)
        x = self.net_activation(x)

    raw_rho = dense_layer(self.num_rho_channels)(x).reshape(
      [-1, num_samples, self.num_rho_channels])  
    return raw_sigma, raw_rho

@jit
def polar2cartesian(theta, phi, radius, origins):
    """ Covert to cartesian coorinates
    Args: 
    theta: [batch, 1]
    phi:   [batch, 1]
    radis: [batch, 1]
    origins: [batch, 3]

      x - x0 = r * sin(theta) * cos (phi)
      y - y0 = r * sin(theta) * sin (phi)
      z - z0 = r * cos(theta)

    Returns:
    (x, y, z) cartesian coorinates
    """
    x = radius * jnp.sin(theta) * jnp.cos(phi) + origins[:, 0]
    y = radius * jnp.sin(theta) * jnp.sin(phi) + origins[:, 1]
    z = radius * jnp.cos(theta) + origins[:, 2]
    return (x, y, z)


def sample_along_hemisphere(key, origins, radius, num_samples, volume_size=None, randomized=False):
  """Sampling along the hemisphere

  Args: 
    key: jnp.ndarray, random generator key.
    origins: jnp.ndarray(float), [batch_size, 3] i.e. camera_grid_positions.
    radius: radius of the hemisphere, [batch_size, 1]
    num_samples: number of samples points.
    volume_size: the bouding box of the volume.

  Returns:
    points: jnp.ndarray, [batch_size, num_samples, 3], sampled_point in the volume space.
  """
  batch_size = origins.shape[0]
  origins = origins.reshape(-1, origins.shape[-1], 1)
  radius = radius.reshape(-1, 1)

  sqrt_samples = int(np.sqrt(num_samples))
  # dtheta, dphi = np.pi / sqrt_samples, np.pi / sqrt_samples
  theta = np.linspace(0, np.pi, sqrt_samples)
  phi = np.linspace(0, np.pi, sqrt_samples)
  theta, phi = np.meshgrid(theta, phi)
  x, y, z = polar2cartesian(theta.flatten(), phi.flatten(), radius, origins)
  
  if DEBUG:
    # plot the sample results
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,32))
    ax = plt.axes(projection='3d')
    ax.scatter(x[0],y[0],z[0], c='b', s=5)
    ax.scatter(*origins[0], c='b', s=5)
    ax.scatter(x[1],y[1],z[1], c='r', s=5)
    ax.scatter(*origins[1], c='r', s=5)
    ax.scatter(x[2],y[2],z[2], c='y', s=5)
    ax.scatter(*origins[2], c='y', s=5)
    plt.savefig('sphere.jpg')
  
  coords = jnp.stack((x, y, z), -1)
  return coords, theta, phi

def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
  """Cat x with positional encoding of x with scales 2^[min_deg, max_deg-1]

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi]
    
  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  if legacy_posenc_order:
    xb = x[Ellipsis, None, :] * scales[:, None]
    four_feat = jnp.reshape(
        jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
        list(x.shape[:-1]) + [-1])
  else:
    xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return jnp.concatenate([x] + [four_feat], axis=-1)





if __name__ == '__main__':
  import pdb
  # model = MLP(
  #       net_depth=8,
  #       net_width=256
  #       )
  # batch = jnp.ones((16, 128, 10))
  # variables = model.init(jax.random.PRNGKey(0), batch)
  # raw_sigma, raw_rho = model.apply(variables, batch)
  # origins = jnp.random.rand((8, 3))
  from datasets import get_dataset
  import pdb
  import argparse
  parser = argparse.ArgumentParser(description="Testing Dataset")
  args = parser.parse_args()
  args.batch_size = 1024*2
  args.data_path = '../data/zaragozadataset/zaragoza256_preprocessed.mat'
  nlos_data = get_dataset(args)
  batch = next(nlos_data)
  origins = batch['grid']
  radius = batch['radius']
  sampled_points = sample_along_hemisphere(jax.random.PRNGKey(0), origins, radius, 64**2)
  pdb.set_trace()
  

