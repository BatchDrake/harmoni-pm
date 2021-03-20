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

from ..transform import CompositeTransform
from ..poasim import POATransform
from ..common import Configuration
from  .fprs_transform import FPRSTransform
from numpy import exp

HARMONI_FPRS_APERTURE   = 1. # Meters
HARMONI_FPRS_TAU        = 0. # Dimensionless
 
class OpticalModel:
    def _extract_params(self):
        self.fprs_atten    = exp(-self.params["fprs.tau"])
        self.fprs_aperture = self.params["fprs.aperture"]
    
    def _init_configuration(self):
        self.params = Configuration()
        
        self.params["fprs.aperture"] = HARMONI_FPRS_APERTURE 
        self.params["fprs.tau"]      = HARMONI_FPRS_TAU
        
        self._extract_params()
        
    def __init__(self):
        self._init_configuration()
        
        self.fprs_transform = FPRSTransform()
        self.poa_transform = POATransform(0, 0)
        self.transform = CompositeTransform()
        
        self.cal_select = True
        
        self.transform.push_back(self.fprs_transform)
        self.transform.push_back(self.poa_transform)
    
    def intensity_to_flux(self):
        return self.fprs_aperture * self.fprs_atten
     
    def load_description(self, path):
        self.params.load(path)
        self._extract_params()
        
    def save_description(self, path):
        self.params.write(path)
        
    def get_transform(self):
        return self.transform
    
    def set_cal(self, cal_select):
        if cal_select is not self.cal_select:
            self.cal_select = cal_select
            # TODO: Add or remove telescope-specific transforms
            pass
    
    def move_to(self, theta, phi):
        self.poa_transform.set_axis_angles(theta, phi)
    
    def generate(self):
        self.transform.generate()