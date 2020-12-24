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

from harmoni_pm.common.array import FloatArray
from harmoni_pm.transform import Transform

class BrownConradyTransform(Transform):
    def __init__(self, k1 = 0, k2 = 0, qx = 0, qy = 0, px = 0, py = 0):
        self.k1 = k1
        self.k2 = k2
        self.qx = qx
        self.qy = qy
        self.p  = FloatArray.make([px, py])
        
    def _forward(self, xy):
        rho    = xy.dot(xy)
        k1rho  = self.k1 * rho
        k2rho2 = self.k2 * rho * rho
        pxy    = self.p.dot(xy)
          
        dx = xy[0] * (k1rho + k2rho2 + pxy) + self.qx * rho
        dy = xy[1] * (k1rho + k2rho2 + pxy) + self.qy * rho    
        
        return FloatArray.make([xy[0] + dx, xy[1] + dy])
    
    def _backward(self, xy):
        # TODO: Implement me
        return xy
    