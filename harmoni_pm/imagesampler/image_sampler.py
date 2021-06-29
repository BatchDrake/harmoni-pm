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
from ..transform import PlaneSampler
from ..common import FloatArray
import numpy as np

HARMONI_IMAGE_SAMPLER_OVERSAMPLING = 8
HARMONI_IMAGE_CENTER_WAVELENGTH    = 500e-9         # m
HARMONI_C                          = 3e8            # s
HARMONI_H                          = 6.62607004e-34 # J s

class ImageSampler(PlaneSampler):
    def _reset_ccd(self):
        self.ccd = np.zeros([self.cols, self.rows])
    
    def _process_region(self, ij, p_xy):
        Ninv = 1. / self.oversampling ** 2
        
        dx = self.finv * self.o_delta_x
        dy = self.finv * self.o_delta_y
         
        p_dx = FloatArray.make([dx, 0.])
        p_dy = FloatArray.make([0., dy])
        
        # TODO: maybe use centered finite differences?
        Tb = self.model.get_transform().backward(self.finv * p_xy)
        dTdx = (Tb - self.model.get_transform().backward(self.finv * p_xy + p_dx)) / dx
        dTdy = (Tb - self.model.get_transform().backward(self.finv * p_xy + p_dy)) / dy
        
        J = dTdx[:, 0] * dTdy[:, 1] - dTdx[:, 1] * dTdy[:, 0]
        
        F = Ninv * self.plane.get_flux(Tb) * J
    
        np.add.at(self.ccd, tuple(ij.transpose()), F)
    
    def __init__(self, plane, model):
        self.plane = plane
        self.model = model
        
        super().__init__()
     
    def precalculate(self):
        super().precalculate()
        self._reset_ccd()

    # Plate scale is always in radians per meter
    def set_detector_geometry(self, cols, rows, delta_x, delta_y, finv = 1):
        super().set_sampling_properties(cols, rows, delta_x, delta_y)
        self.finv = finv

    def integrate(self):
        self._reset_ccd()
        return super().process()
    
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
    