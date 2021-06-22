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

# Lint as: python3
# INTERNAL = False  # pylint: disable=g-statement-before-imports
import json
import os
from os import path
# if not INTERNAL:
# import cv2  # pylint: disable=g-import-not-at-top
import queue
import threading
import numpy as np
import jax
import jax.numpy as jnp
# import jax.scipy as jsp
import scipy.io as scio
import pdb 

__all__ = ["Dataset", "get_dataset"]

def shard(xs):
    """split data into shards for multiple devices along the first dimension."""
    return jax.tree_map(
        lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

def to_device(xs):
    """Transfer data to devices (GPU)"""
    return jax.tree_map(jnp.array, xs)

DEBUG=True

class Dataset(threading.Thread):
    """Dataset Base class"""
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.queue = queue.Queue(3) # set prefetch buffer to 3 batches
        self.daemon = True
        self.batch_size =args.batch_size // jax.process_count()
        self.data_path = args.data_path
        self.batching = "all_grids"
        self._init_datasets()
        self.T, H, W = self.data.shape
        self.data = self.data.reshape(self.T, H * W)
        
        if DEBUG:
            import matplotlib.pyplot as plt
            plt.plot(self.data.sum(-1))
            plt.savefig('test.jpg')

        # Sample tof from 100 to 300
        # should enble auto scaled in the future
        self.sample_Tmin = 100
        self.sample_Tmax = 300
        assert self.sample_Tmin < self.sample_Tmax
        assert self.sample_Tmax <= self.T
        assert self.sample_Tmin >= 0
        self.start()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        x = self.queue.get()
        return shard(x)

    def peek(self):
        """Peek at the next training batch without dequeuing it.
        
        Returns:
            batch: dict, which has grid, hist, index
        """
        x = self.queue.queue[0].copy() # Make a copy of the front of the queue
        return shard(x)

    def run(self): # func scheduled by cpu
        next_func = self._next_train
        while True:
            self.queue.put(next_func())
    
    def _next_train(self):
        """"Sample next training batch"""
        if self.batching == "all_grids": # all_G_all_T
            index_T, index_G = np.random.randint(self.sample_Tmin, self.sample_Tmax, (self.batch_size, )), \
                               np.random.randint(0, self.data.shape[1], (self.batch_size, ))
            sample_hist = self.data[index_T, index_G]
            sample_grid = self.camera_grid_positions[index_G, :]
            sample_radius = index_T * self.c * self.deltaT
            return {"hist": sample_hist, "grid": sample_grid, "radius": sample_radius}

    def _init_datasets(self):
        nlos_data = scio.loadmat(self.data_path)
        self.data = np.array(nlos_data['data'])
        self.deltaT = np.array(nlos_data['deltaT']).item()
        self.camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
        self.camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
        self.camera_grid_positions = np.array(nlos_data['cameraGridPositions']).transpose()
        self.camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
        self.volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
        self.volume_size = np.max(np.array(nlos_data['hiddenVolumeSize']).reshape([-1])).item()
        self.c = 1

def get_dataset(args):
    return Dataset(args)

if __name__ == "__main__":
    import pdb
    import argparse
    parser = argparse.ArgumentParser(description="Testing Dataset")
    args = parser.parse_args()
    args.batch_size = 1024*2
    args.data_path = '../../data/zaragozadataset/zaragoza256_preprocessed.mat'
    nlos_data = Dataset(args)
    batch = next(nlos_data)


