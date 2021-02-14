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

from harmoni_pm.common.prototypes import get_xy
from harmoni_pm.common.exceptions import InvalidTensorShapeError

import numpy as np

#
# Image planes are objects described in the *focal plane*. Focal planes are
# described in terms of size units, not angle units. However, these objects
# are usually sampled in angle units. That's why introduce this focal
# length property, describing how angle sampling is translated into 
# space coordinates in the image plane.
#

class ImagePlane:
    def __init__(self):
        self.f = 1 
        
    def set_focal_length(self, f):
        self.f = f
        pass
    
    def focal_length(self):
        return self.f
    
    def plate_scale(self):
        return 1 / self.f
    
    def _get_intensity(self, xy):
        return 0.
    
    def _get_intensity_matrix(self, matrix):
        if len(matrix.shape) != 2:
            raise InvalidTensorShapeError("High-order tensors not yet supported")

        return np.apply_along_axis(self._get_intensity, 1, matrix)
    
    def get_intensity(self, xy = None, x = None, y = None):
        xy = self.f * get_xy(xy, x, y)
        
        if len(xy.shape) == 1:
            return self._get_intensity(xy)
        else:
            return self._get_intensity_matrix(xy)
