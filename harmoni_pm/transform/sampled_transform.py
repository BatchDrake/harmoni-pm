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

from harmoni_pm.common.array import FloatArray
from harmoni_pm.transform import Transform
from harmoni_pm.common.exceptions import InvalidTensorShapeError

import numpy as np

class SampledTransform(Transform):
    #
    # x0, y0:  Coordinate of the (0, 0) sampling point
    # delta_x: Horizontal step
    # delta_y: Vertical step
    # e_x:     Matrix with as many cols as horizontal sampling points and
    #          as many rows as vertical sampling points, containing the
    #          sampled horizontal displacement.
    # e_y:     Matrix with as many cols as horizontal sampling points and
    #          as many rows as vertical sampling points, containing the
    #          sampled vertical displacement.
    #
    def __init__(self, x0, y0, delta_x, delta_y, e_x, e_y):
        if len(e_x.shape) != 2:
            raise InvalidTensorShapeError("Only 2-D maps are supported")
        elif e_x.shape != e_y.shape:
            raise InvalidTensorShapeError("e_x and e_y shapes are different")
        
        self.p0 = FloatArray.make([x0, y0])
        self.delta = FloatArray.make([delta_x, delta_y])
        
        self.e_x = self._overscan_NE(e_x)
        self.e_y = self._overscan_NE(e_y)
        
        self.rows = e_x.shape[0]
        self.cols = e_x.shape[1]

    def _overscan_NE(self, coef):
        # _overscan_SE: add overscan rows and cols in the north-east directions
        tmpc = np.hstack((coef, np.tile(coef[:, [-1]], 2)))
        return np.vstack((tmpc, np.tile(tmpc[[-1], :], 1)))
    
    def e(self, xy):
        # Calculate the indices belonging to this coordinates, along with
        # their relative offset with respect to the next index (alpha/beta)
        norm    = (xy - self.p0) / self.delta
        indices = np.floor(norm).astype(int)
        
        a = norm - indices
        
        # Some of these coordinates may fall outside the surface covered
        # by the sampling points. These coordinates are easy to spot as the
        # `indices` array will have negative entries or entries >= self.rows
        # or >= self.cols. We identify these valid indices first and leave
        # the displacement of the rest set to zero.  
        
        e = FloatArray.zeros(a.shape)
        
        valid = (0 <= indices[:, 0]) & (indices[:, 0] <= self.cols) & \
                (0 <= indices[:, 1]) & (indices[:, 1] <= self.rows)
                
        alpha = a[valid, 0]
        beta  = a[valid, 1]
        
        i  = indices[valid, 0]
        j  = indices[valid, 1]
        
        e[valid, 0] = (1. - beta) * ((1. - alpha) * self.e_x[j, i] + \
                           alpha * self.e_x[j, i + 1]) + \
                    beta * ((1. - alpha) * self.e_x[j + 1, i] + \
                           alpha * self.e_x[j + 1, i + 1])
        
        e[valid, 1] = (1. - beta) * ((1 - alpha) * self.e_y[j, i] + \
                           alpha * self.e_y[j, i + 1]) + \
                    beta * ((1. - alpha) * self.e_y[j + 1, i] + \
                           alpha * self.e_y[j + 1, i + 1])
        
        
        return e
    
    def _forward_matrix(self, xy):
        return xy + self.e(xy)
    
    def _backward_matrix(self, xy):
        return xy - self.e(xy)
    