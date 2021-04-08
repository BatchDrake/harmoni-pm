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

from .gcu_alignment_transform import GCUAlignmentTransform
from .fprs_transform import FPRSTransform
from .irw_transform import IRWTransform
from .ngss_alignment_transform import NGSSAlignmentTransform

from numpy import exp
from harmoni_pm.poasim.poa_model import POAModel
from harmoni_pm.poasim.poa_center_transform import POACenterTransform

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
        
    def _push_common_transforms(self, c):
        if self.cal_select:
            c.push_back(self.gcu_alignment_transform)
            
        c.push_back(self.fprs_transform)
        c.push_back(self.irw_transform)
        c.push_back(self.ngss_alignment_transform)
        
    def _rebuild_transforms(self):
        # Create composite transforms
        self.transform = CompositeTransform()
        self.pointing_transform = CompositeTransform()
        
        # Build transforms
        self._push_common_transforms(self.transform)
        self.transform.push_back(self.poa_transform)
    
        self._push_common_transforms(self.pointing_transform)
        self.pointing_transform.push_back(self.poa_center_transform)
        
    def __init__(self):
        self._init_configuration()
        
        # Initialize Pick-Off Arm Model
        self.poa_model                = POAModel(None)
        
        # Initialize transforms
        self.gcu_alignment_transform  = GCUAlignmentTransform()
        self.fprs_transform           = FPRSTransform()
        self.irw_transform            = IRWTransform()
        self.ngss_alignment_transform = NGSSAlignmentTransform()
        
        self.poa_transform            = POATransform(self.poa_model)
        self.poa_center_transform     = POACenterTransform(self.poa_model)

        self.cal_select = True
        
        self._rebuild_transforms()
    
    def intensity_to_flux(self):
        return self.fprs_aperture * self.fprs_atten
     
    def load_description(self, path):
        self.params.load(path)
        self._extract_params()
        
    def save_description(self, path):
        self.params.write(path)
        
    def get_transform(self):
        return self.transform
    
    def get_pointing_transform(self):
        return self.pointing_transform
    
    def set_cal(self, cal_select):
        if cal_select is not self.cal_select:
            self.cal_select = cal_select
            self._rebuild_transforms()
            pass
    
    def move_to(self, theta, phi):
        self.poa_transform.set_axis_angles(theta, phi)
    
    def generate(self):
        self.transform.generate()