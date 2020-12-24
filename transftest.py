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

from harmoni_pm.transform import Transform, TransformTester
from harmoni_pm.common import FloatArray
import numpy as np
from harmoni_pm.transform.composite_transform import CompositeTransform
from harmoni_pm.transform.brown_conrady import BrownConradyTransform

class ScaleTransform(Transform):
    def _forward(self, p):
        return 1.1 * p
    
    def _backward(self, p):
        return p / 1.1

class ThirdOrderTransform(Transform):
    def __init__(self, k):
        self.k = k
        
    def _forward(self, p):
        rho = np.linalg.norm(p)
        theta = np.arctan2(p[1], p[0])
            
        rho *= (1 + self.k * rho * rho)
        
        x = rho * FloatArray.make((np.cos(theta), np.sin(theta)))
        return x
    
    def _backward(self, p):
        rho = np.linalg.norm(p)
        theta = np.arctan2(p[1], p[0])
        
        r = (9 * self.k * self.k * rho + 1.7321 * (27 * self.k ** 4 * rho * rho + 4 * self.k ** 3) ** .5)**(1. / 3.)
        
        if np.isnan(r) or r < 1e-13:
            rho = 0
        else:
            rho = 0.38157 * r / self.k - 0.87358 / r
        
        return rho * FloatArray.make((np.cos(theta), np.sin(theta)))
        
class RotationTransform(Transform):
    def __init__(self, theta):
        self.prot = FloatArray.make(
            [[np.cos(theta), np.sin(theta)],
             [-np.sin(theta), np.cos(theta)]])
        self.nrot = FloatArray.make(
            [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]])
        
    def _forward(self, p):
        return self.prot.dot(p)
    
    def _backward(self, p):
        return self.nrot.dot(p)
    
    
def runTest(transf, grid = False):
    transf_name = type(transf).__name__
    print("Running tester on {0}...".format(transf_name))
    tester = TransformTester(transf)

    if grid:
        tester.generate_points(100, 100, 5, 5)
    else:
        tester.generate_stars(85, -1, 100, 100, 6)
        
    tester.sample()
    
    print("  Applying forward...")
    tester.forward()
    print("  Saving...")
    tester.save_to_image(transf_name + "-distorted.png")
    
    print("  Applying backward..")
    tester.backfeed()
    tester.backward()
    
    print("  Saving...")
    tester.save_to_image(transf_name + "-restored.png")
    
    print("  Distortion RMS after undo: {0}".format(tester.distortion_rms()))


runTest(ScaleTransform())
runTest(RotationTransform(np.pi / 180. * 5))
runTest(ThirdOrderTransform(1e-3))

c = CompositeTransform()

c.push_back(ScaleTransform())
c.push_back(RotationTransform(np.pi / 180. * 5))

runTest(c)

runTest(BrownConradyTransform(1e-6, 0, -3e-4, -3e-4, -3e-4, 1e-4), grid = True)
