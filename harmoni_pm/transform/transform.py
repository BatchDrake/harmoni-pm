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

from harmoni_pm.common.exceptions import InvalidPrototype
from harmoni_pm.common.array import FloatArray
import numpy as np

class Transform:
    def get_xy(self, x = None, y = None, xy = None):
        if x is not None and y is not None:
            if not isinstance(x, float) or not isinstance(y, float):
                raise InvalidPrototype("x and y must be floats")
            
            FloatArray.make([x, y])
        elif xy is not None:
            if isinstance(xy, tuple):
                if len(xy) != 2:
                    raise InvalidPrototype("xy must be a tuple of exactly 2 elements")
                elif not isinstance(xy[0], float) or not isinstance(xy[1], float):
                    raise InvalidPrototype("xy elements must be floats")
            
                return FloatArray.make(xy)
            
            elif FloatArray.compatible_with(xy): 
                if xy.shape[len(xy.shape) - 1] != 2:
                    raise InvalidPrototype("Last dimension of xy must be of 2 elements")
                
                return xy
            
            else:
                raise InvalidPrototype("Invalid compound xy coordinates")
        else:
            print(x)
            print(y)
            print(xy)
            raise InvalidPrototype("No coordinates were passed to method")
        
        
    def __forward__(self, xy):
        return xy
    
    def __forward_matrix__(self, matrix):
        if len(matrix.shape) != 2:
            raise InvalidPrototype("High-order tensors not yet supported")
        
        return np.apply_along_axis(self.__forward__, 1, matrix)
    
    def __backward__(self, xy):
        return xy
    
    def __backward_matrix__(self, matrix):
        if len(matrix.shape) != 2:
            raise InvalidPrototype("High-order tensors not yet supported")
        
        return np.apply_along_axis(self.__backward__, 1, matrix)
    
    def forward(self, xy = None, x = None, y = None):
        xy = self.get_xy(x, y, xy)
        
        if len(xy.shape) == 1:
            return self.__forward__(xy)
        else:
            return self.__forward_matrix__(xy)
        
    def backward(self, xy = None, x = None, y = None):
        xy = self.get_xy(x, y, xy)
        
        if len(xy.shape) == 1:
            return self.__backward__(xy)
        else:
            return self.__backward_matrix__(xy)
    
    def generate(self):
        pass
    
    def reset(self):
        pass
    