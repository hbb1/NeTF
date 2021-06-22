# coding=utf-8
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
from typing import Any, Callable
import jax
from flax import linen as nn
from jax import random
import jax.numpy as jnp
from .models_utils import MLP, sample_along_hemisphere, posenc
import pdb

class NeTFModel(nn.Module):
  """NeTF NN Model with both coarse and fine MLPs"""
  num_coarse_samples: int
  num_fine_samples: int
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  net_depth_condition: int = 1  # The depth of the second part of MLP.
  net_width_condition: int = 128  # The width of the second part of MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
  skip_layer: int = 4  # The layer to add skip layers to.
  num_sigma_channels: int = 1  # The number of sigma channels.
  num_rho_channels: int = 1  # The number of rho channels.
  min_deg_point: int = 0
  max_deg_point: int = 10 
  deg_view: int = 5
  legacy_posenc_order: bool=True
  sigma_activation: Callable[Ellipsis, Any] = nn.relu
  rho_activation: Callable[Ellipsis, Any] = nn.relu
  
  def setup(self):
    self.coarse_mlp = MLP(
        net_depth=self.net_depth, 
        net_width=self.net_width,
        net_depth_condition=self.net_depth_condition,
        net_width_condition=self.net_width_condition,
        net_activation=self.net_activation,
        skip_layer=self.skip_layer,
        num_sigma_channels=self.num_sigma_channels,
        num_rho_channels=self.num_rho_channels
    )

  def coarse_encode(self, enc_points, enc_views=None):
    """ implicit representation for the scene using MLP
    Args: 
    enc_points: query coordinates.    [batch_size, num_sample, coord features]
    enc_views: view dependent effects.[batch_size, num_sample, view features]
    Returns:
    raw_sigma: jnp.narray(float32), [batch_size, num_samples, 1]
    raw_rho: jnp.narray(float32), [batch_size, num_samples, 1]
    """
    raw_sigma, raw_rho =  self.coarse_mlp(enc_points, enc_views)
    raw_sigma, raw_rho = self.sigma_activation(raw_sigma), self.rho_activation(raw_rho)
    return raw_sigma, raw_rho

  def predict_hist(self, raw_sigma, raw_rho, theta, phi, radius):
    """ Sum over the hemisphere to obtain the histogram of that trainsiance
    Args:
      raw_sigma: [batch_size, nsamples, 1]
      raw_rho:   [batch_size, nsamples, 1]
      radius:    [batch_size, radius]
      theta:     [nsamples, 1]
      phi:       [nsamples, 1]
      
      \SUM sin(theta) / r^2 * sigma * rho dtheta drho
    
    Returns: 
      histgram [batch_size, 1]
    """
    dtheta, dphi = theta, phi
    dtheta[:, 1:] = theta[:, 1:] - theta[:, :-1]
    dphi[1:, :] = phi[1:, :] - phi[:-1, :]
    # Broadcast condition 
    dtheta, dphi = dtheta.flatten()[None, :, None], dphi.flatten()[None, :, None]
    theta = theta.flatten()[None, :, None]
    radius = radius.flatten()[:, None, None]
    pred_hist = jnp.sin(theta) / radius**2 * raw_sigma * raw_rho * dtheta * dphi
    pred_hist = pred_hist.sum(1)
    return pred_hist

  def __call__(self, rng_0, rng_1, origins, radius, randomized):
    """"NeTT Model.
    Args: 
        rng0: jnp.ndarray, random number generator for coarse model sampling.
        rng1: jnp.ndarray, random number generator for fine model sampling.
        origins: jnp.ndarray [batch_size, 3]
        radius: jnp.ndarray [batch_size, 1]
    Returns: 
        ret: hist
    """
    key, rng_0 = random.split(rng_0)
    # sampling_along_sphere 
    origins = origins.reshape(-1, 3)
    coords, theta, phi = sample_along_hemisphere(key,
                                                origins,
                                                radius,
                                                self.num_coarse_samples)
    batch_size = coords.shape[0]
    # views = origins[:, None, :] - coords
    views = jnp.concatenate((theta.reshape(-1, 1), phi.reshape(-1,1)), axis=-1)
    views = views[None, Ellipsis].repeat(batch_size, 0)

    enc_points = posenc(coords, self.min_deg_point, self.max_deg_point, legacy_posenc_order=True)
    enc_views = posenc(views, 0, self.deg_view, legacy_posenc_order=True)
    raw_sigma, raw_rho = self.coarse_encode(enc_points, enc_views)
    # coords : batch * nsamples
    # hemisphere: batch
    pred_hist = self.predict_hist(raw_sigma, raw_rho, theta, phi, radius)
    return (pred_hist, )


def construct_netf(key, example_batch, args):
  """ Construct a Neural Trainsiance Field
  Args:
    key: jnp.ndarray. Random number generator.
    example_batch: dict, an example of batch of data
    args: FLAGS class. 
  Returns: 
    models: nn.Module. Merf model with parameters.
    states: flax.Module.state, Nerf model state for stateful parameters.
  """
  net_activation = getattr(nn, str(args.net_activation))
  rho_activation = getattr(nn, str(args.rho_activation))
  sigma_activation = getattr(nn, str(args.sigma_activation))

  x = jnp.exp(jnp.linspace(-90, 90, 1024))
  x = jnp.concatenate([-x[::-1], x], 0)
  rho = rho_activation(x)
  if jnp.any(rho < 0):
  # or jnp.any(rho > 1):
    raise NotImplementedError(
      "Choice of rho_activation `{}` produces reflacteness outside [0,1]"
      .format(args.rho_activation)
    )
  sigma = sigma_activation(x)
  if jnp.any(sigma < 0):
    raise NotImplementedError(
      "Choice of sigma_activation `{}` produces negative densities"
      .format(args.rho_activation)
    )
  model = NeTFModel(
    min_deg_point=args.min_deg_point,
    max_deg_point=args.max_deg_point,
    deg_view=args.deg_view,
    num_coarse_samples=args.num_coarse_samples, 
    num_fine_samples=args.num_fine_samples,
    sigma_activation=sigma_activation,
    rho_activation=rho_activation,
    net_activation=net_activation
  )
  key1, key2, key3 = random.split(key, num=3)
  init_variables = model.init(key1, 
                    key2, key3, 
                    example_batch['grid'],
                    example_batch['radius'],
                    False)
  return model, init_variables


if __name__ == '__main__':
  import pdb
  model = NeTFModel(
              num_coarse_samples=16*16,
              num_fine_samples=64*64,
          )
  from datasets import get_dataset

  import pdb
  import argparse
  parser = argparse.ArgumentParser(description="Testing Dataset")
  args = parser.parse_args()
  args.batch_size = 1024*2
  args.data_path = '../data/zaragozadataset/zaragoza256_preprocessed.mat'
  args.min_deg_point = 0
  args.max_deg_point = 10
  args.deg_view = 4
  args.net_activation = "relu"
  args.rho_activation = "sigmoid"
  args.sigma_activation = "relu"
  args.num_coarse_samples = 16*16
  args.num_fine_samples = 32*32
  nlos_data = get_dataset(args)
  batch = next(nlos_data)
  origins = batch['grid']
  radius = batch['radius']
  key = jax.random.PRNGKey(0)
  model, variables = construct_netf(key, example_batch=batch, args=args)
