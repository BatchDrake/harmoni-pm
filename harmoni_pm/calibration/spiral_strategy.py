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

from harmoni_pm.common import Configuration
from harmoni_pm.common import FloatArray
from harmoni_pm.calibration import CalibrationStrategy
from harmoni_pm.calibration import CalibrationStrategyFactory
from harmoni_pm.calibration import CalibrationStrategyCollection

import numpy as np

HARMONI_SPIRAL_STRATEGY_NUMBER = 11

class SpiralStrategy(CalibrationStrategy):
    def _init_params(self):
        self.params = Configuration()
        
        self.params["cal.number"] = HARMONI_SPIRAL_STRATEGY_NUMBER
         
    def _extract_params(self):
        self.N = self.params["cal.number"]
        if self.N < 0:
            raise ValueError("Too few calibration points")
        
        
    def __init__(self, gcu, config):
        self.gcu = gcu
        self._init_params()
        self.params.copy_from(config)
        self._extract_params()
        self.alpha = 2 * np.pi * np.random.uniform(0, 1)

    def generate_points(self, scale = 1.):
        P = []
        
        if self.N == 1:
            P = [[0., 0.]]
        else:
            Nc = np.floor(np.sqrt(self.N / np.pi))
            theta_max = 2 * np.pi * Nc
            dtheta =  theta_max / (self.N - 1)
            for j in range(0, self.N):
                rho   = scale * np.sqrt(j / float(self.N - 1))
                theta = self.alpha + j * dtheta
                
                P.append([rho * np.cos(theta), rho * np.sin(theta)])
        
        calpoints = FloatArray.make(P)
        
        return self.gcu.closest(self.gcu.unnormalize(calpoints))
    
class SpiralStrategyFactory(CalibrationStrategyFactory):
    @staticmethod
    def register():
        try:
            CalibrationStrategyCollection().register(
                "spiral",
                SpiralStrategyFactory())
        except:
            pass
        
    def make(self, gcu, config):
        return SpiralStrategy(gcu, config)

