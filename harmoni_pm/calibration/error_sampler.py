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

from harmoni_pm.transform import PlaneSampler
from harmoni_pm.common import FloatArray

import numpy as np

class ErrorSampler(PlaneSampler):
    def __init__(self, transform):
        super().__init__()
        
        self.transform = transform
        self.err_xy  = FloatArray.make(np.zeros([0, 2]))
        self.err_abs = FloatArray.make(np.zeros([0]))
        
    def _process_region(self, ij, xy):
        Ninv = 1. / self.oversampling ** 2
        
        self.err_xy = xy - self.transform.backward(xy)
        self.err_abs = np.linalg.norm(self.err_xy, axis = 1)
        
        ij[:, 1] = self.rows - ij[:, 1] - 1
         
        np.add.at(self.err_map, tuple(ij.transpose()), Ninv * self.err_abs)
        
    def reset_err_map(self):
        self.err_map = np.zeros([self.cols, self.rows])
        
    def precalculate(self):
        super().precalculate()
        self.reset_err_map()
        
    def process_points(self, points):
        self.reset_err_map()
        
        prev_oversampling = self.oversampling
        self.oversampling = 1
        result = super().process_points(points)
        self.oversampling = prev_oversampling
        
        return (self.err_xy, result[0], result[1])
    
    def get_error_vec(self):
        return self.err_xy
    
    def get_error_abs(self):
        return self.err_abs
    
    def get_error_map(self):
        return self.err_map
    
    