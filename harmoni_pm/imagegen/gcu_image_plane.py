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
from harmoni_pm.imagegen.image_plane import ImagePlane
from harmoni_pm.common.array import FloatArray

HARMONI_GCU_POINT_SEPARATION = 15e-3  # m
HARMONI_GCU_POINT_DIAMETER   = 150e-6 # m
HARMONI_GCU_POINT_FLUX       = 1e-3   # W Hz^-1 m^-2
HARMONI_GCU_MASK_X0          = 0      # m
HARMONI_GCU_MASK_Y0          = 0      # m
HARMONI_GCU_MASK_DIAMETER    = 400e-3 # m

class GCUImagePlane(ImagePlane):        
    def _extract_params(self):
        self.p_sep      = self.params["gcu.point.separation"]
        self.p_diam     = self.params["gcu.point.diameter"]
        self.p_int      = self.params["gcu.point.flux"]
        self.m_diameter = self.params["gcu.mask.diameter"]
        
        self.m_p0       = FloatArray.make(
            [self.params["gcu.mask.x0"], self.params["gcu.mask.y0"]])

    def set_params(self, params = None):
        if params is not None:
            self.params.copy_from(params)
            
        self._extract_params()
                
    def _init_params(self):
        self.params = Configuration()
        
        self.params["gcu.point.separation"] = HARMONI_GCU_POINT_SEPARATION
        self.params["gcu.point.diameter"]   = HARMONI_GCU_POINT_DIAMETER
        self.params["gcu.point.flux"]       = HARMONI_GCU_POINT_FLUX
        self.params["gcu.mask.x0"]          = HARMONI_GCU_MASK_X0
        self.params["gcu.mask.y0"]          = HARMONI_GCU_MASK_Y0
        self.params["gcu.mask.diameter"]    = HARMONI_GCU_MASK_DIAMETER
       
        self.skip  = None
        self.R_int = None
        
    def _in_radius_interval(self, xy, a, b):
        radii = np.linalg.norm(
            np.round((xy - self.m_p0) / self.p_sep) * self.p_sep,
            axis = 1)
        return (a <= radii) & (radii < b)
    
    def in_mask(self, xy): 
        return self._in_radius_interval(xy, 0, .5 * self.m_diameter)
    
    def unnormalize(self, xy = None, x = None, y = None):
        uxy = get_xy(xy, x, y)

        return .5 * uxy * self.m_diameter + self.m_p0
    
    def closest(self, xy = None, x = None, y = None):
        xy = get_xy(xy, x, y)
        
        c = self.p_sep * np.round((xy - self.m_p0) / self.p_sep) + self.m_p0
        
        if len(c.shape) == 1:
            c = np.reshape(c, (1, 2))
            
        return c[self.in_mask(c), :]
    
    def point_list(self, skip = 0, R_int = None):
        if skip < 0:
            raise ValueError("Invalid point skip value")
        
        if self.skip != skip or self.R_int != R_int:
            self.skip  = skip
            self.R_int = R_int
            
            step   = skip + 1
            sep    = self.p_sep * step
            points = int(np.ceil(.5 * self.m_diameter / sep))
            coords = []
            
            if R_int is None:
                min_R = 0
                max_R = .5 * self.m_diameter
            elif type(R_int) is float or type(R_int) is int:
                min_R  = 0
                max_R  = R_int
            elif type(R_int) is tuple:
                min_R  = R_int[0]
                max_R  = R_int[1]
            else:
                raise ValueError("Invalid radius interval")
            
            for j in range(-points, points):
                for i in range(-points, points):
                    x = i * sep + self.m_p0[0]
                    y = j * sep + self.m_p0[1]
                    p = [x, y]
                    
                    if self._in_radius_interval([p], min_R, max_R)[0]:
                        coords.append(p)
            
            self.last_points = FloatArray.make(coords)
            
        return self.last_points
    
    def __init__(self, params = None):
        super().__init__()
        self._init_params()
        self.set_params(params)
        
    def generate(self):
        pass
    
    def save_to_file(self, path):
        self.params.write(path)
        
    def load_from_file(self, path):
        self.params.load(path)
        self._extract_params()
    
    def _get_flux_matrix(self, xy):
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
            
        return self.p_int      \
            * self.in_mask(xy) \
            * (np.linalg.norm(mod, axis = 1) <= .5 * self.p_diam)
    
    def _get_flux(self, xy):
        return self.get_flux_from_matrix(xy)

        
    