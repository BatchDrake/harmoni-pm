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

from .complex import ComplexZernike
from ..common.exceptions import InvalidTensorShapeError 
import numpy as np

class ZernikeSolver:
    def __init__(self, points, J = 10):
        self.delta = points
        self.J = J
        basis = []
        
        if len(points.shape) != 2:
            raise InvalidTensorShapeError(
                "Invalid point matrix shape (must be Nx2)")
        
        if False and points.shape[0] < J:
            raise InvalidTensorShapeError(
                "Too few points to solve {0} Zernike coefficients".format(J))
            
        for j in range(J):
            m, n = ComplexZernike.j_to_mn(j)
            Z    = ComplexZernike.Z(m, n)
            
            # Z(points) is basically equivalent to <Z, sum delta_k>
            basis.append(Z(points))
        
        self.Z = np.array(basis).transpose()
        
    def solve_for(self, err):
        if len(err.shape) != 2:
            raise InvalidTensorShapeError(
                "Invalid tensor shape (must be {0}x2)".format(
                    self.delta.shape[0]))
        
        if (self.delta.shape[0] != err.shape[0]) or (self.delta.shape[1] != err.shape[1]): 
            raise InvalidTensorShapeError(
                "Expecting an error matrix of {0}x{1} coefficients (got {2}x{3} instead)".format(
                    self.delta.shape[0],
                    self.delta.shape[1],
                    err.shape[0],
                    err.shape[1]))
        
        c_e = err[:, 0] + 1j * err[:, 1]
        
        ret = np.linalg.lstsq(self.Z, c_e, rcond = None)
        
        return ret[0]
    
        