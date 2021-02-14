#
# Copyright (c) 2021 Gonzalo J. Carracedo <BatchDrake@gmail.com>
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
import numpy as np
from harmoni_pm.common import Configuration
from harmoni_pm.common.prototypes import get_xy
from harmoni_pm.common.exceptions import InvalidTensorShapeError
from harmoni_pm.imagegen.image_plane import ImagePlane
from harmoni_pm.common.array import FloatArray

HARMONI_GCU_POINT_SEPARATION = 15e-3  # m
HARMONI_GCU_POINT_DIAMETER   = 150e-6 # m
HARMONI_GCU_POINT_INTENSITY  = 1e-3   # W Hz^-1 m^-2 sr^-1
HARMONI_GCU_MASK_X0          = 0      # m
HARMONI_GCU_MASK_Y0          = 0      # m
HARMONI_GCU_MASK_DIAMETER    = 400e-3 # m

class GCUImagePlane(ImagePlane):        
    def _extract_params(self):
        self.p_sep      = self.params.get("point.separation")
        self.p_diam     = self.params.get("point.diameter")
        self.p_int      = self.params.get("point.intensity")
        self.m_p0       = FloatArray.make(
            [self.params.get("mask.x0"), self.params.get("mask.y0")])
        self.m_diameter = self.params.get("mask.diameter")
        
    def __init__(self, params = None):
        super().__init__()
        self.params = Configuration()
        
        self.params.set("point.separation", HARMONI_GCU_POINT_SEPARATION)
        self.params.set("point.diameter",   HARMONI_GCU_POINT_DIAMETER)
        self.params.set("point.intensity",  HARMONI_GCU_POINT_INTENSITY)
        self.params.set("mask.x0",          HARMONI_GCU_MASK_X0)
        self.params.set("mask.y0",          HARMONI_GCU_MASK_Y0)
        self.params.set("mask.diameter",    HARMONI_GCU_MASK_DIAMETER)
     
        if params is not None:
            self._copy_params(params)
            
        self._extract_params()
        
    def generate(self):
        pass
    
    def save_to_file(self, path):
        self.params.write(path)
        
    def load_from_file(self, path):
        self.params.load(path)
        self._extract_params()
    
    def _get_intensity(self, xy):
        mod = np.fmod(
            1.5 * self.p_sep + np.fmod(xy - self.m_p0, self.p_sep), 
            self.p_sep) - .5 * self.p_sep

        # TODO: Add diffuse light?
        
        #
        # Protip: never (ever) return something here different from a floating
        # point number. Many C programmers (me among them) tend to use short
        # literals like 0 and expect it to be automatically promoted to the
        # right type, either at compile time or at runtime. This is a bad
        # habit of mine that ended up consuming a full Saturday in order to
        # figure out a bug.
        #   
        # Long story short: if this function is called from numpy.apply_along_axis,
        # and the first value it returns is a 0 (and not 0.), apply_along_axis
        # will assume you are building an integer matrix and round all floating
        # point numbers returned next.
        #
        
        I = self.p_int if np.linalg.norm(mod) <= .5 * self.p_diam else 0.
    
        return I
    
    def set_params(self, params):
        self.parse_dict(params)
        self._extract_params()
        
    