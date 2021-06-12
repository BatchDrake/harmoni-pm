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
from ..common.configuration import Configuration
from ..tolerance import GQ
from scipy.sparse import bsr_matrix
from harmoni_pm.common.array import FloatArray
from harmoni_pm.transform import Transform, ZernikeTransform

HARMONI_POA_ENCODER_BITS     = 11
HARMONI_POA_QUANTIZATION_ERR = "0.5 +/- 0.5 dimensionless (flat)" # units w.r.t level
HARMONI_POA_RADIUS           = "0.2 +/- 1e-6 m (flat)"
HARMONI_POA_ARM_INSTABILITY  = "0.0 +/- 1e-6 m (gauss)"
HARMONI_POA_POSITION_ERROR   = "0.0 +/- 0 m (gauss)"

HARMONI_POA_ENCODER_ERROR    = "0.0 +/- 1 arcsec (flat)"

class POAModel:
    def _init_params(self):
        self.params = Configuration()
        
        self.params["poa.encoder[theta].bits"]  = HARMONI_POA_ENCODER_BITS
        self.params["poa.encoder[theta].qerr"]  = HARMONI_POA_QUANTIZATION_ERR 
        self.params["poa.encoder[theta].error"] = HARMONI_POA_ENCODER_ERROR
        
        self.params["poa.encoder[phi].bits"]    = HARMONI_POA_ENCODER_BITS
        self.params["poa.encoder[phi].qerr"]    = HARMONI_POA_QUANTIZATION_ERR
        self.params["poa.encoder[phi].error"]   = HARMONI_POA_ENCODER_ERROR
        
        self.params["poa.radius"]               = HARMONI_POA_RADIUS
        self.params["poa.arm_instability"]      = HARMONI_POA_ARM_INSTABILITY
        self.params["poa.position_error"]       = HARMONI_POA_POSITION_ERROR
        
    def generate(self, event = "manufacture"):
        if event == "manufacture":
            # Nominal radius. This is the arm length requested to the 
            # manufacturer and the one used for coordinate transform.
            self.R   = self.radius.value("m")
            
            # Manufacture-time radius. This is the true arm length
            # delivered by the manufacturer.
            self.m_R = self.radius.generate(1, "m")
    
    def _extract_params(self):
        self.pos_error       = GQ(self.params["poa.position_error"])
        self.radius          = GQ(self.params["poa.radius"])
        self.arm_instability = GQ(self.params["poa.arm_instability"])
        
        self.step_count      = [
            2 ** self.params["poa.encoder[theta].bits"],
            2 ** self.params["poa.encoder[phi].bits"]]
        
        self.q_error         = [
            GQ(self.params["poa.encoder[theta].qerr"]),
            GQ(self.params["poa.encoder[phi].qerr"])]
        
        self.enc_error       = [
            GQ(self.params["poa.encoder[theta].error"]),
            GQ(self.params["poa.encoder[phi].error"])]
        
        self.generate("manufacture")
        
    def set_params(self, params = None):
        if params is not None:
            self.params.copy_from(params)
            
        self._extract_params()
        
    def set_error_model(self, model):
        if model is None:
            self.corrective_transform = Transform()
        else:
            self.corrective_transform = ZernikeTransform(model, self.R)
        
    def __init__(self, params = None):
        self.corrective_transform = Transform()
        self._init_params()
        self.set_params(params)
        
    def xy_to_theta_phi(self, xy, mirror = False):
        sign    = -1 if mirror else 1
        rho     = np.linalg.norm(xy, axis = 1)
        alpha   = np.arctan2(xy[:, 1], xy[:, 0])
        cos_phi = 1 - .5 * rho ** 2 / self.R ** 2
        phi     = sign * np.arccos(cos_phi)
        beta    = np.arctan2(np.sin(phi), 1 - cos_phi)
        theta   = alpha - beta
        
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
        twopi = 2 * np.pi
        theta_phi  = FloatArray.make(theta_phi)
        
        if len(theta_phi.shape) == 1:
            theta_phi = np.reshape(theta_phi, (int(len(theta_phi) / 2), 2))
            
        n          = theta_phi.shape[0]
        theta_phi /= twopi
        
        theta_phi -= np.floor(theta_phi)
        
        digital_theta = (
            np.floor(theta_phi[:, 0] * self.step_count[0]) 
            + self.q_error[0].generate(n)) / self.step_count[0] \
            + self.enc_error[0].generate(n, "radians") / twopi
            
        digital_phi   = (
            np.floor(theta_phi[:, 1] * self.step_count[1]) 
            + self.q_error[1].generate(n)) / self.step_count[1] \
            + self.enc_error[1].generate(n, "radians") / twopi
        
        return FloatArray.make(
            twopi * np.column_stack((digital_theta, digital_phi)))
    
    def model_xy_from_theta_phi(self, theta_phi):
        q_theta_phi = self.model_theta_phi(theta_phi)
        theta       = q_theta_phi[:, 0]
        diff        = q_theta_phi[:, 1] - q_theta_phi[:, 0]
        n           = len(theta)
        R           = self.m_R + self.arm_instability.generate(n, "meters")
        x           = self.m_R * np.cos(theta) - R * np.cos(diff)
        y           = self.m_R * np.sin(theta) + R * np.sin(diff)
        
        # Model generic positioning error (usually gaussian)
        p_e_r       = self.pos_error.generate(n, "meters")
        
        # Uniform distribution of directions. Important: samples are uniformly
        # distributed over the half-open interval [0, 2pi). This condition is
        # necessary to prevent a bias towards the positive x direction. 
        p_e_alpha   = np.random.uniform(0, 2 * np.pi, n)
        
        x += p_e_r * np.cos(p_e_alpha)
        y += p_e_r * np.sin(p_e_alpha)
        
        return np.column_stack((x, y))

    def model_xy(self, xy, mirror = False): 
        return self.model_xy_from_theta_phi(
            self.xy_to_theta_phi(
                self.corrective_transform.backward(xy), 
                mirror))
