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
from harmoni_pm.zernike import ZernikeSolver
from .error_sampler import ErrorSampler

import numpy as np

HARMONI_CALIBRATION_MAX_J = 10                 # 1
HARMONI_CALIBRATION_BEARING_GAP = 10e-3        # m

HARMONI_CALIBRATION_SAMPLER_ROWS = 400         # 1
HARMONI_CALIBRATION_SAMPLER_COLS = 400         # 1
HARMONI_CALIBRATION_STEP_X       = 2.5 * .5e-3 # m
HARMONI_CALIBRATION_STEP_Y       = 2.5 * .5e-3 # m


class Calibration:
    def __init__(
            self, 
            config, 
            J = HARMONI_CALIBRATION_MAX_J, 
            gap = HARMONI_CALIBRATION_BEARING_GAP):
        self.model = OpticalModel(config)
        self.gcu   = GCUImagePlane(config)
        self.gcu_points = self.gcu.point_list()
        self.J     = J
        
        # TODO: Get field diameter
        self.solver = ZernikeSolver(self.gcu_points / self.model.R(), J)

        self.pointing_transform = self.model.get_pointing_transform()
        
        self.sampler = ErrorSampler(self.pointing_transform)
        
        self.sampler.set_sampling_properties(
            HARMONI_CALIBRATION_SAMPLER_COLS, 
            HARMONI_CALIBRATION_SAMPLER_ROWS, 
            HARMONI_CALIBRATION_STEP_X, 
            HARMONI_CALIBRATION_STEP_Y, 
            radius = self.model.R() - gap)

    def get_axes(self):
        return [self.sampler.xmin(), 
                self.sampler.xmax(), 
                self.sampler.ymin(), 
                self.sampler.ymax()]
            
    def manufacture(self):
        self.model.generate("manufacture")
        pass
    
    def start_session(self):
        self.model.generate("session")
        pass
    
    def get_gcu_points(self, up_to = None):
        if up_to is not None:
            indices = np.arange(self.gcu_points.shape[0])
            np.random.shuffle(indices)
            return self.gcu_points[indices[0:up_to], :]
        
        return self.gcu_points
    
    def set_pointing_model(self, params):
        self.model.set_pointing_model(params)
        
    def measure_displacements(self, points = None):
        if points is None:
            points = self.get_gcu_points()
            
        self.sampler.process_points(points)
        self.err_sq = self.sampler.get_error_sq()
        self.err_vec = self.sampler.get_error_vec()

    def get_error_map(self):
        return self.sampler.get_error_map()
    
    def sample_error_map(self):
        self.sampler.reset_err_map()
        self.sampler.process()
        
        self.err_sq = self.get_error_map().flatten()
        
    def solve_pointing_model(self):
        return self.solver.solve_for(self.err_vec)
        
    def test_model(self, params, full = False):
        # TODO: Disable instabilities!!
        self.set_pointing_model(params)
        if full:
            self.sample_error_map()
        else:
            self.measure_displacements()
        self.set_pointing_model(None)

    def get_mse(self, params, full = False):
        self.test_model(params, full)
        return np.mean(self.err_sq)

    def get_max_se(self, params, full = False):
        self.test_model(params, full)
        return np.max(self.err_sq)

    def calibrate(self, points = None):
        if points is None:
            points = self.get_gcu_points()
        
        # Step 1: start a calibration session
        self.start_session()
        
        # Step 2: measure all displacements
        self.measure_displacements(points)
        
        # Step 3: solve pointing model
        solver = ZernikeSolver(points / self.model.R(), self.J)
        return solver.solve_for(self.err_vec)
    
    def sample_pointing_model(self, count = 100):
        i = 0
        a = np.zeros([count, self.J], dtype = 'complex128')
        
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