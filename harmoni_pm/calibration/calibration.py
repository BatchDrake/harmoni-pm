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
from .calibration_strategy_collection import CalibrationStrategyCollection
from harmoni_pm.common.exceptions import InvalidTensorShapeError
from harmoni_pm.common.array import FloatArray

import numpy as np

HARMONI_CALIBRATION_MAX_J = 10                 # 1
HARMONI_CALIBRATION_BEARING_GAP = 10e-3        # m

HARMONI_CALIBRATION_SAMPLER_ROWS = 400         # 1
HARMONI_CALIBRATION_SAMPLER_COLS = 400         # 1
HARMONI_CALIBRATION_STEP_X       = 2.5 * .5e-3 # m
HARMONI_CALIBRATION_STEP_Y       = 2.5 * .5e-3 # m
HARMONI_CALIBRATION_STRATEGY     = "random"


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
        
        self.rho_scale = (self.model.R() - gap) / self.model.R()
        
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
    
    def generate_points(self, number, strategy = "random"):
        return CalibrationStrategyCollection().generate_points(
            self.gcu,
            strategy,
            {"cal.number" : number},
            scale = self.rho_scale)
        
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
    
    def _sort_points_by_axis_distance(
            self, 
            points, 
            exp = np.inf, 
            random = False):
        
        if random:
            points = points.copy()
            s = np.random.randint(points.shape[0])
            a, b = points[0, :].copy(), points[s, :].copy()
            points[0, :], points[s, :] = b, a
        
        theta_phi = self.model.poa_model.xy_to_theta_phi(points)

        new_points = [points[0, :].tolist()]
        for i in range(1, points.shape[0]):
            p     = theta_phi[i - 1, :]  # Reference point
            delta = theta_phi[i:, :] - p # Angle deltas
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            dists = np.linalg.norm(delta, axis = 1, ord = exp)
            min_i = i + np.argmin(dists)
            
            # Swap!
            a, b = theta_phi[i, :].copy(), theta_phi[min_i, :].copy()
            theta_phi[i, :], theta_phi[min_i, :] = b, a
            
            new_points.append(
                self.model.poa_model.xy_from_theta_phi(
                    theta_phi[i:i + 1, :]).flatten().tolist())
            
        
        result = FloatArray.make(new_points)
        return result
    
    def get_calibration_path(
            self, 
            points = None, 
            t_num = 50, 
            t_cal = 0,
            optimize = False,
            exponent = np.inf,
            random = False):
        if points is None:
            points = self.get_gcu_points()
                
        if len(points.shape) != 2:
            raise InvalidTensorShapeError(
                "Invalid point matrix shape (must be Nx2)")
    
        if optimize:
            points = self._sort_points_by_axis_distance(
                points, 
                exp = exponent,
                random = random)
            
        N = points.shape[0]
        
        # Step 1: start a calibration session
        self.start_session()
        
        # Step 2: compute t_num timesteps per point pair
        poa_model = self.model.poa_model
        if N < 2:
            return (
                FloatArray.make([0]),
                self.points, 
                poa_model.model_theta_phi_from_xy(self.points))
        
        # Step 3:  
        t, alpha = poa_model.model_sweep(points[0:-1], points[1:], t_num)
        
        final_theta = []
        final_phi   = []
        final_t     = []
        t_p         = []
        t0          = 0
        
        # Shape of both t and alpha is t_num x N x 2. For each angle we will
        # have different time samplings. We will construct two interpolators
        # to obtain a combined time sampling axis for both axes
        for i in range(t.shape[1]):
            t_p.append(t0)
            t_pos_max = np.max(t[-1, i, :])
            t_combined = np.linspace(0, t_pos_max, t_num)
            
            # Note that one of both axes will finish positioning earlier
            # than the other. We encode this situation by adding an extra
            # time step in the end at the time all axes have finished
            # positioning.
            
            t_theta = np.append(t[:, i, 0], t_pos_max)
            t_phi   = np.append(t[:, i, 1], t_pos_max)
            
            theta   = np.append(alpha[:, i, 0], alpha[-1, i, 0])
            theta   = np.interp(t_combined, t_theta, theta)
            
            phi     = np.append(alpha[:, i, 1], alpha[-1, i, 1])
            phi     = np.interp(t_combined, t_phi, phi)
            
            final_t.append(t0 + t_combined)
            final_theta.append(theta)
            final_phi.append(phi)
            
            t0     += t_pos_max + t_cal
            
        t_p.append(t0)
        final_theta = FloatArray.make(final_theta).flatten()
        final_phi   = FloatArray.make(final_phi).flatten()
        
        theta_phi   = FloatArray.make([final_theta, final_phi]).transpose()
        final_t     = FloatArray.make(final_t).flatten()
        
        xy          = poa_model.xy_from_theta_phi(theta_phi)
        
        return (final_t, theta_phi, xy, t_p, points)
    
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