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

from harmoni_pm.imagegen import GCUImagePlane
from harmoni_pm.optics import OpticalModel
from harmoni_pm.zernike import ComplexZernike, ZernikeSolver

from .error_sampler import ErrorSampler

import numpy as np

class Calibration:
    def __init__(self, config, J = 3):
        self.model = OpticalModel(config)
        self.gcu   = GCUImagePlane(config)
        self.gcu_points = self.gcu.point_list()
        self.J     = J
        
        # TODO: Get field diameter
        self.solver = ZernikeSolver(self.gcu_points / self.model.R(), J)

        self.pointing_transform = self.model.get_pointing_transform()
        
        self.sampler = ErrorSampler(self.pointing_transform)
        
        self.sampler.set_sampling_properties(
            400, 
            400, 
            2.5 * .5e-3, 
            2.5 * .5e-3, 
            radius = self.model.R())

    def manufacture(self):
        self.model.generate("manufacture")
        pass
    
    def start_session(self):
        self.model.generate("session")
        pass
    
    def set_pointing_model(self, params):
        self.model.set_pointing_model(params)
        
    def measure_displacements(self):
        self.sampler.process_points(self.gcu_points)
        self.err_abs = self.sampler.get_error_abs()
        self.err_vec = self.sampler.get_error_vec()
         
    def solve_pointing_model(self):
        return self.solver.solve_for(self.err_vec)
        
    def get_mse(self, params):
        # TODO: Disable instabilities!!
        self.set_pointing_model(params)
        self.measure_displacements(self.gcu_points)
        
        return np.sum(self.err_abs ** 2)

    def sample_pointing_model(self, count = 100):
        i = 0
        a = np.zeros([count, self.J], dtype = 'complex64')
        
        while i < count:
            self.manufacture()
            self.measure_displacements()
            a[i, :] = self.solve_pointing_model()
            i += 1
            if i % 100 == 0:
                print("Sampling: {0:5}/{1:5} done\r".format(i, count), end = '')
                
        return a
    
    # TODO: Run multiple resolutions of the Zernike polynomials and return
    # a posterior. Optionally, accept a prior.