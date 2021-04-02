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
from ..common import FloatArray
from ..tolerance import GenerativeQuantity

import numpy as np


HARMONI_POA_FIELD_RADIUS  = 0.4 # 400mm field
HARMONI_POA_ARM_RADIUS    = 0.4 # 400mm radius

HARMONI_POA_ENCODER_BITS  = 10  # bits
HARMONI_POA_POSITION_OFFSET = "0.5+/-0.5 dimensionless (flat)" # units w.r.t level

class POATransform(Transform):
    def __init__(self, theta, phi):
        self.step_count = 2 ** HARMONI_POA_ENCODER_BITS
        self.pos_off    = GenerativeQuantity(HARMONI_POA_POSITION_OFFSET)
        
        self.set_axis_angles(theta, phi)
    
    def quantize_theta_phi(self, theta, phi):
        theta /= 2 * np.pi
        phi   /= 2 * np.pi
        
        theta -= np.floor(theta)
        phi   -= np.floor(phi)
        
        digital_theta = (
            np.floor(theta * self.step_count) + self.pos_off.generate()) / self.step_count
             
        digital_phi   = (
            np.floor(phi   * self.step_count) + self.pos_off.generate()) / self.step_count
        
        return (digital_theta * 2 * np.pi, digital_phi * 2 * np.pi)
    
    def set_axis_angles(self, theta, phi):
        theta, phi = self.quantize_theta_phi(theta, phi)
        
        self.theta   = theta
        self.phi     = phi
        self.rotangl = phi - theta
        self.center  = FloatArray.make([
            HARMONI_POA_FIELD_RADIUS * np.cos(self.theta) - HARMONI_POA_ARM_RADIUS * np.cos(self.rotangl),
            HARMONI_POA_FIELD_RADIUS * np.sin(self.theta) + HARMONI_POA_ARM_RADIUS * np.sin(self.rotangl)])
        
        self.fwd_rot = FloatArray.make(
            [[np.cos(self.rotangl),  -np.sin(self.rotangl)],
             [np.sin(self.rotangl), np.cos(self.rotangl)]])
        
        self.bwd_rot = np.transpose(self.fwd_rot)
        
    def get_cost(self):
        return 0
    
    def reset_cost(self):
        pass
    
    def _forward_matrix(self, p):
        return self.fwd_rot.dot((p - self.center).transpose())
    
    def _forward(self, p):
        return self.fwd_rot.dot(p - self.center)
    
    def _backward_matrix(self, p):
        return self.bwd_rot.dot(p.transpose()).transpose() + self.center
    
    def _backward(self, p):
        return self.bwd_rot.dot(p) + self.center

    