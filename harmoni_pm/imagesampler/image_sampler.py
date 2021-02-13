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
import numpy as np

HARMONI_IMAGE_SAMPLER_OVERSAMPLING = 8

class ImageSampler:
    def _reset_ccd(self):
        self.ccd = np.zeros([self.cols, self.rows])
    
    def _integrate_pixel(self, p_i, p_j):
        px   = p_i * self.delta_x + self.x0 
        py   = p_j * self.delta_y + self.y0
        odx   = self.delta_x / self.oversampling
        ody   = self.delta_y / self.oversampling
        Ninv = 1. / (self.oversampling * self.oversampling)
        
        #
        # Sampling from CCD: using backward transformations
        # TODO: Intensity is defined in energy units per unit time, so 
        # in order to get true count, we must first define
        #
        # 1: Integration time
        # 2: Frequency (so we can get true photon count)
        # 
        # Also, this is actually a Poisson process. We must actually drag these
        # counts from a Poisson distribution. 
        #
        
        j = 0
        while j < self.oversampling:
            i = 0
            while i < self.oversampling:
                self.ccd[p_j, p_i] += \
                    Ninv * self.plane.get_intensity(
                        self.model.get_transform().backward(
                                x = self.finv * (px + i * odx), 
                                y = self.finv * (py + j * ody))) 
        
                i += 1
            j += 1
            
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
        self._reset_ccd()
        
    def integrate(self):
        self._reset_ccd()
        j = 0
        while j < self.rows:
            print("{0:.1f}%".format(np.floor(1000 * j / (self.rows - 1)) * .1), end = "\r") 
            i = 0
            while i < self.cols:
                self._integrate_pixel(i, j)
                i += 1
            j += 1
    
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
        