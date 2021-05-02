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

from ..transform import Transform 

import numpy as np

class POATransform(Transform):
    def __init__(self, model):
        self.model = model
        self.set_axis_angles(0, 0)

    def set_axis_angles(self, theta, phi):
        m_thetaphi = self.model.model_theta_phi((theta, phi))
        
        self.center = self.model.xy_from_theta_phi(m_thetaphi)
        self.fwd_rot = self.model.R_from_theta_phi(m_thetaphi)
            
        self.bwd_rot = np.transpose(self.fwd_rot)
        
    def get_cost(self):
        return 0
    
    def reset_cost(self):
        pass
    
    def generate(self, event = "manufacture"):
        self.model.generate(event)
    
    def _forward_matrix(self, p):
        return self.fwd_rot.dot((p - self.center).transpose()).transpose()
    
    def _forward(self, p):
        return self.fwd_rot.dot(p - self.center)
    
    def _backward_matrix(self, p):
        return self.bwd_rot.dot(p.transpose()).transpose() + self.center
    
    def _backward(self, p):
        return self.bwd_rot.dot(p) + self.center

    