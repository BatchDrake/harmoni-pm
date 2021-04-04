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

from harmoni_pm.transform import PlaneSampler
from harmoni_pm.optics   import OpticalModel
from harmoni_pm.common import FloatArray

import matplotlib.pyplot as plt

import argparse, sys
import numpy as np
from harmoni_pm.common.configuration import Configuration

POINTINGSIM_DEFAULT_FIELD_DIAMETER = 4e-1 # m
POINTINGSIM_DEFAULT_FIELD_DELTA_X  = 1e-3 # m 
POINTINGSIM_DEFAULT_FIELD_DELTA_Y  = 1e-3 # m

class PointingSimulator(PlaneSampler):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.model  = OpticalModel()
        
        self.pointing_transform = self.model.get_pointing_transform()
        
    def _reset_measures(self):
        self.model.move_to(0, 0)
        
        self.measures = np.zeros([self.cols, self.rows])
        
    def _process_region(self, ij, xy):
        Ninv = 1. / self.oversampling ** 2
        
        err = np.linalg.norm(xy - self.pointing_transform.backward(xy), axis = 1)
        
        ij[:, 1] = self.rows - ij[:, 1] - 1
         
        np.add.at(self.measures, tuple(ij.transpose()), Ninv * err)
        
    def precalculate(self):
        super().precalculate()
        self._reset_measures()
    
    def plot(self):
        axes = FloatArray.make(
            [self.xmin(), self.xmax(), self.ymin(), self.ymax()])
        
        plt.imshow(
            1e3 * self.measures.transpose(), 
            cmap = plt.get_cmap("inferno"),
            extent = 1e3 * axes)
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        
        c = plt.colorbar()
        c.set_label("Pointing error (mm)")
        
    def show(self):
        plt.show()
        
ps = PointingSimulator(Configuration)
ps.set_sampling_properties(400, 400, 1e-4, 1e-4, radius = 0.2)

for i in range(4):
    row = i / 2
    col = i % 2
    print("Drawing", i)
    plt.subplot(2, 2, i + 1)
    
    ps._reset_measures()
    ps.process()
    ps.plot()
    
ps.show()

