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

from harmoni_pm.common.exceptions import InvalidTensorShapeError
from harmoni_pm.common.prototypes import get_xy

import numpy as np

class Transform:        
    def _forward(self, xy):
        return xy
    
    def _forward_matrix(self, matrix):
        if len(matrix.shape) != 2:
            raise InvalidTensorShapeError("High-order tensors not yet supported")
        
        return np.apply_along_axis(self._forward, 1, matrix)
    
    def _backward(self, xy):
        return xy
    
    def _backward_matrix(self, matrix):
        if len(matrix.shape) != 2:
            raise InvalidTensorShapeError("High-order tensors not yet supported")
        
        return np.apply_along_axis(self._backward, 1, matrix)
    
    def forward(self, xy = None, x = None, y = None):
        xy = get_xy(xy, x, y)
        
        if len(xy.shape) == 1:
            return self._forward(xy)
        else:
            return self._forward_matrix(xy)
        
    def backward(self, xy = None, x = None, y = None):
        xy = get_xy(xy, x, y)
        
        if len(xy.shape) == 1:
            return self._backward(xy)
        else:
            return self._backward_matrix(xy)
    
    def generate(self):
        pass
    
    def reset(self):
        pass
    