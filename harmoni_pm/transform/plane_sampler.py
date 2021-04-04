#
# Copyright (c) 2020 Gonzalo J. Carracedo <BatchDrake@gmail.com>
# 
#
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this 
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, 
#    this list of conditions and the following disclaimer in the documentation 
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors 
#    may be used to endorse or promote products derived from this software 
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from ..common import FloatArray
from ..common.exceptions import AbstractClassCallError
import concurrent.futures

import numpy as np
import time

HARMONI_PLANE_SAMPLER_SLICE_SIZE   = 128
HARMONI_PLANE_SAMPLER_OVERSAMPLING = 8

class PlaneSampler:
    def __init__(self):
        self.oversampling = HARMONI_PLANE_SAMPLER_OVERSAMPLING
        self.set_parallel(True)
        
    def _process_slice(self, coords):
        start = time.time()
        
        i_start = coords[0]
        j_start = coords[1]
        
        i_end = min(i_start + HARMONI_PLANE_SAMPLER_SLICE_SIZE, self.cols)
        j_end = min(j_start + HARMONI_PLANE_SAMPLER_SLICE_SIZE, self.rows)
        
        i_len = i_end - i_start
        j_len = j_end - j_start
        
        i = np.linspace(i_start, i_end - 1, i_len) *  np.ones([j_len, 1])
        j = (np.linspace(j_start, j_end - 1, j_len) *  np.ones([i_len, 1])).transpose()
        
        # This is a size x 2 array containing all the pixel indices of the
        # CCD.
        
        ij = np.array([i.flatten(), j.flatten()]).transpose().reshape(
            i_len * j_len, 
            2).astype(int)
        
        # The total number of coordinates will be ij.rows() x oversampling^2
        # We can achieve this by repeating the coordinates inside ij 
        # oversampling^2 times, and tiling the subpixel offsets ij.rows() times
        
        p_xy = np.tile(self.xy, (ij.shape[0], 1))
        
        # Compute the position of the top left corner of each pixel
        ij = np.repeat(ij, self.oversampling ** 2, 0)
        
        # The full coordinate list is now just p_xy + o_xy
        p_xy += ij * [self.delta_x, self.delta_y] + [self.x0, self.y0]
        
        # Radius limit defined. Remove cells outside this radius
        if self.radius is not None:
            subset = np.linalg.norm(p_xy, axis = 1) <= self.radius
            ij   = ij[subset,   :]
            p_xy = p_xy[subset, :]
            
        #
        # Process region. We provide both the cell indices and the coordinates
        # inside each cell.
        #
        
        if len(ij) > 0:
            self._process_region(ij, p_xy)
        
        return time.time() - start
    
    def set_sampling_properties(self, cols, rows, delta_x, delta_y, radius = None):
        self.cols     = cols
        self.rows     = rows
        self.delta_x  = delta_x
        self.delta_y  = delta_y
        self.x0       = -.5 * cols * delta_x
        self.y0       = -.5 * rows * delta_y
        self.radius   = radius
        
        self.precalculate()
        
    def xmin(self):
        return self.x0
    
    def xmax(self):
        return self.x0 + self.cols * self.delta_x
    
    def ymin(self):
        return self.y0
    
    def ymax(self):
        return self.y0 + self.rows * self.delta_y
    
    def set_parallel(self, val):
        self.parallel = val
        
    def precalculate(self):
        osp = np.linspace(0, self.oversampling - 1, self.oversampling) * \
            np.ones([self.oversampling, 1])
        
        self.o_delta_x  = self.delta_x / self.oversampling
        self.o_delta_y  = self.delta_y / self.oversampling
        
        x = self.o_delta_x * (osp.flatten() + .5)
        y = self.o_delta_y * (osp.transpose().flatten() + .5)
    
        self.betaA = self.model.intensity_to_flux()
        
        self.xy = FloatArray.make([x, y]).transpose()
        
    def set_oversampling(self, oversampling):
        if oversampling != self.oversampling:
            self.oversampling = oversampling
            self.precalculate()
        
    
    def process_parallel(self):
        slices = []
        # Prepare list of slices
        # TODO: remove loops
        j = 0
        while j < self.rows:
            i = 0
            while i < self.cols:
                slices.append((i, j))
                i += HARMONI_PLANE_SAMPLER_SLICE_SIZE
            j += HARMONI_PLANE_SAMPLER_SLICE_SIZE
        
        # Spawn threads
        with concurrent.futures.ThreadPoolExecutor(max_workers = 20) as executor:
            self.delays = list(executor.map(self._process_slice, slices))
            
    
    def process_serial(self):
        j = 0
        while j < self.rows:
            i = 0
            while i < self.cols:
                self.delays.append(self._process_slice((i, j)))
                i += HARMONI_PLANE_SAMPLER_SLICE_SIZE
            j += HARMONI_PLANE_SAMPLER_SLICE_SIZE
        
    def process(self):
        self.delays = []
        
        execution_start = time.time()
    
        if self.parallel:
            self.process_parallel()
        else:
            self.process_serial()
            
        execution_end = time.time()
        # Return performance figures
        return (
            len(self.delays), 
            np.mean(self.delays), 
            np.std(self.delays), 
            execution_end - execution_start)
    
    def _process_region(self, ij, xy):
        raise AbstractClassCallError("Calling a virtual method in abstract class")
    
    