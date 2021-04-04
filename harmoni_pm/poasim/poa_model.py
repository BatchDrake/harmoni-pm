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
from ..tolerance import GenerativeQuantity
from scipy.sparse import bsr_matrix
from harmoni_pm.common.array import FloatArray

HARMONI_POA_ENCODER_BITS    = 11  # bits

HARMONI_POA_POSITION_OFFSET = "0.5 +/- 0.5 dimensionless (flat)" # units w.r.t level
HARMONI_POA_ARM_LENGTH      = "0.4 +/- 1e-5 m"

class POAModel:
    def __init__(self, config):
        self.set_config(config)
        
    def set_config(self, config):
        # TODO: actually take these values from POA configuration
        
        self.config = config
        
        self.step_count = 2 ** HARMONI_POA_ENCODER_BITS
        self.pos_off    = GenerativeQuantity.make(HARMONI_POA_POSITION_OFFSET)
        self.arm_length = GenerativeQuantity.make(HARMONI_POA_ARM_LENGTH)
        self.R          = self.arm_length["meters"]
        
    def xy_to_theta_phi(self, xy, mirror = False):
        sign    = -1 if mirror else 1
        rho     = np.linalg.norm(xy, axis = 1)
        alpha   = np.arctan2(xy[:, 1], xy[:, 0])
        cos_phi = 1 - .5 * rho ** 2 / self.R ** 2
        phi     = sign * np.arccos(cos_phi)
        beta    = np.arctan2(1 - cos_phi, np.sin(phi))
        theta   = phi + alpha - (beta + .5 * np.pi)
        
        return np.column_stack((theta, phi))
    
    def xy_from_theta_phi(self, theta_phi):
        theta       = theta_phi[:, 0]
        diff        = theta_phi[:, 1] - theta
        x           = self.R * (np.cos(theta) - np.cos(diff))
        y           = self.R * (np.sin(theta) + np.sin(diff))
        
        return np.column_stack((x, y))

    def R_from_theta_phi(self, theta_phi):
        #
        # This computes the forward matrices of the theta_phi coordinate list
        #
        # If we have 128 x 64 = 8192 (theta, phi) pairs, we would need to
        # compose a (8192 * 2)^2 = 10^6 element matrix, whose diagonals are
        # made out of individual rotation matrices, the rest of it being
        # zero. In cases like this, we better use a sparse representation
        # of the matrix to reduce memory footprint 
        rotangl = theta_phi[:, 1] - theta_phi[:, 0]
        n       = len(rotangl)
        
        # The best representation is a block-sparse matrix. We need to provide
        # an array of individual matrices, an array of column indices and
        # an array of pew-row column index ranges.
        
        indices = range(n)         # Column index of each row
        indptr  = range(n + 1)     # Positions of the column indices
        
        data = np.zeros((n, 2, 2)) # Nonzero data: N 2x2 rotation matrices
        
        data[:, 0, 0] =  np.cos(rotangl)
        data[:, 0, 1] = -np.sin(rotangl)
        data[:, 1, 0] = -data[:, 0, 1]
        data[:, 1, 1] =  data[:, 0, 0]
        
        return bsr_matrix((data, indices, indptr), dtype = 'float32')
    
    def model_theta_phi(self, theta_phi):
        theta_phi  = FloatArray.make(theta_phi)
        
        if len(theta_phi.shape) == 1:
            theta_phi = np.reshape(theta_phi, (int(len(theta_phi) / 2), 2))
            
        n          = theta_phi.shape[0]
        theta_phi /= 2 * np.pi
        
        theta_phi -= np.floor(theta_phi)
        
        digital_theta = (
            np.floor(theta_phi[:, 0] * self.step_count) + self.pos_off.generate(n)) / self.step_count
             
        digital_phi   = (
            np.floor(theta_phi[:, 1]  * self.step_count) + self.pos_off.generate(n)) / self.step_count
        
        return 2 * np.pi * np.column_stack((digital_theta, digital_phi))
    
    def model_xy_from_theta_phi(self, theta_phi):
        q_theta_phi = self.model_theta_phi(theta_phi)
        theta       = q_theta_phi[:, 0]
        diff        = q_theta_phi[:, 1] - q_theta_phi[:, 0]
        n           = len(theta)
        R           = self.arm_length.generate(n)
        x           = self.R * np.cos(theta) - R * np.cos(diff)
        y           = self.R * np.sin(theta) + R * np.sin(diff)
        
        return np.column_stack((x, y))

    def model_xy(self, xy, mirror = False):
        return self.model_xy_from_theta_phi(self.xy_to_theta_phi(xy, mirror))
        