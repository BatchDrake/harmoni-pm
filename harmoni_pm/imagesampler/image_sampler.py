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
import numpy as np

HARMONI_IMAGE_SAMPLER_OVERSAMPLING = 8

class ImageSampler:
    def _reset_ccd(self):
        self.ccd = np.zeros([self.cols, self.rows])
    
    def _integrate_pixel(self, ij):
        Ninv = 1. / self.oversampling ** 2
        
        # The total number of coordinates will be ij.rows() x oversampling^2
        # We can achieve this by repeating the coordinates inside ij 
        # oversampling^2 times, and tiling the subpixel offsets ij.rows() times
        
        p_xy = np.tile(self.xy, (ij.shape[0], 1))
        
        # Compute the position of the top left corner of each pixel
        ij = np.repeat(ij, self.oversampling ** 2, 0)
        
        p_xy += self.finv * ij * [self.delta_x, self.delta_y] + [self.x0, self.y0]

        # The full coordinate list is now just p_xy + o_xy
        I = self.plane.get_intensity(self.model.get_transform().backward(p_xy))
    
        np.add.at(self.ccd, tuple(ij.transpose()), Ninv * I)
        
    def __init__(self, plane, model):
        self.plane        = plane
        self.model        = model
        self.oversampling = HARMONI_IMAGE_SAMPLER_OVERSAMPLING
        
        
    # Plate scale is always in radians per unit length
    def set_detector_geometry(self, cols, rows, delta_x, delta_y, finv = 1):
        self.cols    = cols
        self.rows    = rows
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.x0      = -.5 * cols * delta_x
        self.y0      = -.5 * rows * delta_y
        self.finv    = finv
        
        osp = np.linspace(0, self.oversampling - 1, self.oversampling) * \
            np.ones([self.oversampling, 1])
        
        odx  = self.delta_x / self.oversampling
        ody  = self.delta_y / self.oversampling
        
        x = self.finv * osp.flatten() * odx
        y = self.finv * osp.transpose().flatten() * ody
    
        self.xy = FloatArray.make([x, y]).transpose()
        
        self._reset_ccd()
        
    def integrate(self):
        self._reset_ccd()
        
        i = np.linspace(0, self.cols - 1, self.cols)  * \
            np.ones([self.rows, 1])
        j = (np.linspace(0, self.rows - 1, self.rows) * \
            np.ones([self.cols, 1])).transpose()
        
        # This is a size x 2 array containing all the pixel indices of the
        # CCD.
        
        ij = np.array([i.flatten(), j.flatten()]).transpose().reshape(
            self.rows * self.cols, 
            2).astype(int)
        
        self._integrate_pixel(ij)
        
    def save_to_file(self, path):
        # This square root is for representation purposes only. CCD stores 
        # something that is proportional to the gathered energy, while
        # image information refers to the amplitude of the wave. The
        # relationship between both is precisely a square root.
        
        ccdsqrt = np.sqrt(self.ccd)
        maxv = ccdsqrt.max()       
        if np.abs(maxv) > 0:
            k = 255 / maxv
        else:
            k = 1
        
        im = Image.fromarray((k * ccdsqrt).astype(np.uint8))
        
        im.save(path)
        