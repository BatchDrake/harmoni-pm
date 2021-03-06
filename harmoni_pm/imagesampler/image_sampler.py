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

from PIL import Image
from ..common import FloatArray
import concurrent.futures
import numpy as np
import time

HARMONI_IMAGE_SAMPLER_SLICE_SIZE   = 128

HARMONI_IMAGE_SAMPLER_OVERSAMPLING = 8
HARMONI_IMAGE_CENTER_WAVELENGTH    = 500e-9         # m
HARMONI_C                          = 3e8            # s
HARMONI_H                          = 6.62607004e-34 # J s

class ImageSampler:
    def _reset_ccd(self):
        self.ccd = np.zeros([self.cols, self.rows])
    
    def _integrate_pixels(self, ij):
        Ninv = 1. / self.oversampling ** 2
        
        dx = self.delta_x / self.oversampling
        dy = self.delta_y / self.oversampling
         
        p_dx = self.finv * FloatArray.make([dx, 0.])
        p_dy = self.finv * FloatArray.make([0., dy])
        
        # The total number of coordinates will be ij.rows() x oversampling^2
        # We can achieve this by repeating the coordinates inside ij 
        # oversampling^2 times, and tiling the subpixel offsets ij.rows() times
        
        p_xy = np.tile(self.xy, (ij.shape[0], 1))
        
        # Compute the position of the top left corner of each pixel
        ij = np.repeat(ij, self.oversampling ** 2, 0)
        
        p_xy += self.finv * (
            ij * [self.delta_x, self.delta_y] + [self.x0, self.y0])
        # The full coordinate list is now just p_xy + o_xy
        
        # Numerical calculation of the Jacobian, according to the current
        # oversampling configuration. It is assumed that the oversampling
        # already provides information about the sub-Nyquist structure of the
        # image plane.
        
        # TODO: maybe use centered finite differences?
        Tb = self.model.get_transform().backward(p_xy)
        dTdx = (Tb - self.model.get_transform().backward(p_xy + p_dx)) / dx
        dTdy = (Tb - self.model.get_transform().backward(p_xy + p_dy)) / dy
        
        J = dTdx[:, 0] * dTdy[:, 1] - dTdx[:, 1] * dTdy[:, 0]
        
        F = Ninv * self.betaA * self.plane.get_intensity(Tb) * J
    
        np.add.at(self.ccd, tuple(ij.transpose()), F)
    
    def __init__(self, plane, model):
        self.plane        = plane
        self.model        = model
        self.oversampling = HARMONI_IMAGE_SAMPLER_OVERSAMPLING
        
        self.set_parallel(True)
     
    def precalculate(self):
        osp = np.linspace(0, self.oversampling - 1, self.oversampling) * \
            np.ones([self.oversampling, 1])
        
        odx  = self.delta_x / self.oversampling
        ody  = self.delta_y / self.oversampling
        
        x = self.finv * osp.flatten() * odx + .5 * odx
        y = self.finv * osp.transpose().flatten() * ody + .5 * ody
    
        self.betaA = self.model.intensity_to_flux()
        
        self.xy = FloatArray.make([x, y]).transpose()
        
        self._reset_ccd()
        
    def set_oversampling(self, oversampling):
        if oversampling != self.oversampling:
            self.oversampling = oversampling
            self.precalculate()
           
    # Plate scale is always in radians per meter
    def set_detector_geometry(self, cols, rows, delta_x, delta_y, finv = 1):
        self.cols     = cols
        self.rows     = rows
        self.delta_x  = delta_x
        self.delta_y  = delta_y
        self.x0       = -.5 * cols * delta_x
        self.y0       = -.5 * rows * delta_y
        self.finv     = finv
        
        self.precalculate()
        
    def set_parallel(self, val):
        self.parallel = val
            
    def _integrate_slice(self, coords):
        start = time.time()
        
        i_start = coords[0]
        j_start = coords[1]
        
        i_end = min(i_start + HARMONI_IMAGE_SAMPLER_SLICE_SIZE, self.cols)
        j_end = min(j_start + HARMONI_IMAGE_SAMPLER_SLICE_SIZE, self.rows)
        
        i_len = i_end - i_start
        j_len = j_end - j_start
        
        i = np.linspace(i_start, i_end - 1, i_len)  *  np.ones([i_len, 1])
        j = (np.linspace(j_start, j_end - 1, j_len) *  np.ones([j_len, 1])).transpose()
        
        # This is a size x 2 array containing all the pixel indices of the
        # CCD.
        
        ij = np.array([i.flatten(), j.flatten()]).transpose().reshape(
            i_len * j_len, 
            2).astype(int)
        
        self._integrate_pixels(ij)
        
        return time.time() - start

    def integrate_parallel(self):
        slices = []
        
        # Prepare list of slices
        # TODO: remove loops
        j = 0
        while j < self.rows:
            i = 0
            while i < self.cols:
                slices.append((i, j))
                i += HARMONI_IMAGE_SAMPLER_SLICE_SIZE
            j += HARMONI_IMAGE_SAMPLER_SLICE_SIZE
        
        # Spawn threads
        with concurrent.futures.ThreadPoolExecutor(max_workers = 20) as executor:
            self.delays = list(executor.map(self._integrate_slice, slices))
        
    def integrate_serial(self):
        j = 0
        while j < self.rows:
            i = 0
            while i < self.cols:
                self.delays.append(self._integrate_slice((i, j)))
                i += HARMONI_IMAGE_SAMPLER_SLICE_SIZE
            j += HARMONI_IMAGE_SAMPLER_SLICE_SIZE
        
        
    def integrate(self):
        self._reset_ccd()
        self.delays = []
        
        execution_start = time.time()
    
        if self.parallel:
            self.integrate_parallel()
        else:
            self.integrate_serial()
            
        execution_end = time.time()
        # Return performance figures
        return (
            len(self.delays), 
            np.mean(self.delays), 
            np.std(self.delays), 
            execution_end - execution_start)
    
    def save_to_file(self, path, delta_t = None, linear = False):
        # This square root is for representation purposes only. CCD stores 
        # something that is proportional to the gathered energy, while
        # image information refers to the amplitude of the wave. The
        # relationship between both is precisely a square root.
       
        if delta_t is not None:
            Ap = self.delta_x * self.delta_y
            
            # TODO: Calculate Q appropriately, don't assume wavelengths
            f2q = HARMONI_IMAGE_CENTER_WAVELENGTH / (HARMONI_H * HARMONI_C)
            
            print(
                "Peak photon flux: {0} / (s px)".format(
                    np.max(f2q * self.ccd * Ap * delta_t)))
            ccd = np.random.poisson(f2q * self.ccd * Ap * delta_t) 
        else:
            ccd = self.ccd
            
        if not linear:
            ccd = np.sqrt(ccd)
            
        maxv = ccd.max()       
        if np.abs(maxv) > 0:
            k = 255 / maxv
        else:
            k = 1
        
        im = Image.fromarray((k * ccd).astype(np.uint8))
        
        im.save(path)
    